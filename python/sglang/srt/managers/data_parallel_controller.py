# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A controller that dispatches requests to multiple data parallel workers."""

import faulthandler
import logging
import multiprocessing as mp
import signal
import threading
import time
from enum import Enum, auto
from typing import Callable, List, Optional

import psutil
import setproctitle
import zmq

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import compute_dp_attention_world_info
from sglang.srt.managers.io_struct import (
    ActiveRanksOutput,
    BlockReqInput,
    DPMigrateReq,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
    WatchLoadUpdateReq,
)
from sglang.srt.managers.schedule_batch import Req, RequestStage
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.metrics.cpu_monitor import start_cpu_monitor_thread
from sglang.srt.server_args import (
    DP_ATTENTION_HANDSHAKE_PORT_DELTA,
    PortArgs,
    ServerArgs,
)
from sglang.srt.tracing.trace import (
    process_tracing_init,
    trace_get_proc_propagate_context,
    trace_set_proc_propagate_context,
    trace_set_thread_info,
    trace_slice_end,
    trace_slice_start,
)
from sglang.srt.utils import numa_utils
from sglang.srt.utils.common import (
    bind_port,
    configure_ipv6,
    configure_logger,
    get_zmq_socket,
    kill_itself_when_parent_died,
    maybe_reindex_device_id,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.watchdog import Watchdog
from sglang.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


class LoadBalanceMethod(Enum):
    """Load balance method."""

    ROUND_ROBIN = auto()
    FOLLOW_BOOTSTRAP_ROOM = auto()
    TOTAL_REQUESTS = auto()
    TOTAL_TOKENS = auto()

    @classmethod
    def from_str(cls, method: str):
        method = method.upper()
        try:
            return cls[method]
        except KeyError as exc:
            raise ValueError(f"Invalid load balance method: {method}") from exc


class DPBudget:
    def __init__(self, dp_size: int):
        self.dp_size = dp_size
        self.total_requests = [0] * dp_size
        self.total_tokens = [0] * dp_size
        # Per-rank lightest request info for migration
        self.lightest_req_rid = [None] * dp_size
        self.lightest_req_tokens = [None] * dp_size

    def update_budget(self, load_update: WatchLoadUpdateReq):
        """Update the budget."""
        for load in load_update.loads:
            self.total_requests[load.dp_rank] = load.num_reqs
            self.total_tokens[load.dp_rank] = load.num_tokens
            self.lightest_req_rid[load.dp_rank] = load.lightest_req_rid
            self.lightest_req_tokens[load.dp_rank] = load.lightest_req_tokens

    def dispatch(self, method: LoadBalanceMethod):
        if method == LoadBalanceMethod.TOTAL_REQUESTS:
            target_rank = self.total_requests.index(min(self.total_requests))
        elif method == LoadBalanceMethod.TOTAL_TOKENS:
            # Use total_requests as a tie-breaker when total_tokens are equal
            target_rank = min(
                range(self.dp_size),
                key=lambda i: (self.total_tokens[i], self.total_requests[i]),
            )
        else:
            return None

        # Increment the load of that worker by one as a heuristic
        self.total_requests[target_rank] += 1
        return target_rank


class DPMigrationBalancer:
    """Implements the minimum-weight migration algorithm for DP load balancing.

    On each load update tick:
    1. Rank DP ranks by total token load.
    2. Identify the top-M heaviest and bottom-M lightest ranks.
    3. Find the lightest (youngest) request across all heavy ranks.
    4. Trigger migration if mean_heavy - mean_light > alpha * w_min.
    5. Migrate the lightest request from its heavy rank to the lightest rank.
    6. Enter cooldown for tau ticks to prevent oscillation.
    """

    def __init__(self, dp_size: int, alpha: float = 2.0, top_m: int = 4,
                 cooldown_seconds: float = 10.0):
        self.dp_size = dp_size
        self.alpha = alpha
        self.top_m = top_m
        self.cooldown_seconds = cooldown_seconds
        self._last_migration_time = 0.0
        # Track rids currently being migrated to avoid double-migration
        self._pending_migrations: set = set()
        # Track recently migrated rids to prevent ping-pong
        # Maps rid -> timestamp when migration completed
        self._recently_migrated: dict = {}
        # Per-rid cooldown: don't re-migrate a request within this window
        self._rid_cooldown_seconds = cooldown_seconds * 6  # 60s default

    def check(self, budget: DPBudget):
        """Check for imbalance and return migration actions.

        Returns:
            List of (rid, src_rank, dst_rank) tuples to migrate.
        """
        now = time.monotonic()
        if now - self._last_migration_time < self.cooldown_seconds:
            return []

        if self.dp_size < 2:
            return []

        # 1. Rank by load (total tokens)
        rank_loads = [
            (i, budget.total_tokens[i]) for i in range(self.dp_size)
        ]
        rank_loads.sort(key=lambda x: x[1])

        m = min(self.top_m, self.dp_size // 2)
        if m == 0:
            return []

        light = rank_loads[:m]
        heavy = rank_loads[-m:]

        mean_heavy = sum(load for _, load in heavy) / m
        mean_light = sum(load for _, load in light) / m

        # 2. Find lightest request across heavy ranks
        best_rid = None
        best_tokens = None
        best_src_rank = None
        for rank, _ in heavy:
            rid = budget.lightest_req_rid[rank]
            tokens = budget.lightest_req_tokens[rank]
            if rid is None or tokens is None:
                continue
            if rid in self._pending_migrations:
                continue
            # Don't migrate internal health check requests
            if rid.startswith("HEALTH_CHECK_"):
                continue
            # Don't re-migrate recently migrated requests (prevents ping-pong)
            if rid in self._recently_migrated:
                if now - self._recently_migrated[rid] < self._rid_cooldown_seconds:
                    continue
                else:
                    del self._recently_migrated[rid]
            if best_tokens is None or tokens < best_tokens:
                best_tokens = tokens
                best_rid = rid
                best_src_rank = rank

        if best_rid is None or best_tokens is None:
            return []

        # 3. Trigger condition
        w_min = best_tokens
        if mean_heavy - mean_light <= self.alpha * w_min:
            return []

        # 4. Pick lightest destination rank
        dst_rank = light[0][0]

        logger.info(
            f"DPMigrationBalancer: triggering migration rid={best_rid} "
            f"src_rank={best_src_rank} dst_rank={dst_rank} "
            f"w_min={w_min} mean_heavy={mean_heavy:.0f} mean_light={mean_light:.0f} "
            f"gap={mean_heavy - mean_light:.0f} threshold={self.alpha * w_min:.0f}"
        )

        self._pending_migrations.add(best_rid)
        self._last_migration_time = time.monotonic()

        return [(best_rid, best_src_rank, dst_rank)]

    def migration_completed(self, rid: str):
        """Called when a migration completes (success or failure)."""
        self._pending_migrations.discard(rid)
        self._recently_migrated[rid] = time.monotonic()


class DataParallelController:
    """A controller that dispatches requests to multiple data parallel workers."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        run_scheduler_process_func: Callable,
    ) -> None:
        # Parse args
        self.server_args = server_args
        self.port_args = port_args
        self.load_balance_method = LoadBalanceMethod.from_str(
            server_args.load_balance_method
        )
        self.run_scheduler_process_func = run_scheduler_process_func

        # For DP balance
        self.global_balance_id = 0

        # Init inter-process communication
        self.context = zmq.Context(2 + server_args.dp_size)
        if server_args.node_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                self.context, zmq.PULL, port_args.scheduler_input_ipc_name, False
            )

        # Dispatch method
        self.round_robin_counter = 0
        dispatch_lookup = {
            LoadBalanceMethod.ROUND_ROBIN: self.round_robin_scheduler,
            LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM: self.follow_bootstrap_room_scheduler,
            LoadBalanceMethod.TOTAL_REQUESTS: self.total_requests_scheduler,
            LoadBalanceMethod.TOTAL_TOKENS: self.total_tokens_scheduler,
        }
        self.dispatching = dispatch_lookup[self.load_balance_method]

        # Load balance budget
        self.dp_budget = DPBudget(server_args.dp_size)

        # DP migration balancer
        self.migration_balancer = None
        if getattr(server_args, "enable_dp_migration", False):
            self.migration_balancer = DPMigrationBalancer(
                dp_size=server_args.dp_size,
                alpha=getattr(server_args, "dp_migration_alpha", 2.0),
                top_m=getattr(server_args, "dp_migration_top_m", 4),
                cooldown_seconds=getattr(server_args, "dp_migration_cooldown", 10.0),
            )
            # PUSH socket to send DPMigrateReq to the tokenizer manager
            # (connects to the same PULL socket the tokenizer listens on)
            if server_args.node_rank == 0:
                self.send_to_tokenizer = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    port_args.tokenizer_ipc_name,
                    False,
                )
            logger.info(
                f"DP migration balancer enabled: alpha={server_args.dp_migration_alpha}, "
                f"top_m={server_args.dp_migration_top_m}, "
                f"cooldown={server_args.dp_migration_cooldown}"
            )

        # To protect changing env vars to set CUDA_VISIBLE_DEVICES.
        self.env_lock = threading.Lock()

        # Launch data parallel workers
        self.scheduler_procs = []
        self.workers: List[zmq.Socket] = [None] * server_args.dp_size
        self.status: List[bool] = [True] * server_args.dp_size

        if server_args.enable_dp_attention:
            self.launch_dp_attention_schedulers(server_args, port_args)
            self.control_message_step = server_args.tp_size
        else:
            self.launch_dp_schedulers(server_args, port_args)
            self.control_message_step = 1

        self.init_dispatcher()

        self.soft_watchdog = Watchdog.create(
            debug_name="DataParallelController",
            watchdog_timeout=server_args.soft_watchdog_timeout,
            soft=True,
            test_stuck_time=envs.SGLANG_TEST_STUCK_DP_CONTROLLER.get(),
        )

        if server_args.enable_metrics:
            start_cpu_monitor_thread("data_parallel_controller")

    def send_to_all_workers(self, obj):
        for i, worker in enumerate(self.workers):
            if self.status[i]:
                worker.send_pyobj(obj)

    def send_control_message(self, obj):
        # Send control messages to first worker of tp group
        for worker in self.workers[:: self.control_message_step]:
            worker.send_pyobj(obj)

    def handle_load_update_req(self, obj):
        self.dp_budget.update_budget(obj)

        # Check for migration triggers
        if self.migration_balancer is not None:
            migrations = self.migration_balancer.check(self.dp_budget)
            for rid, src_rank, dst_rank in migrations:
                cmd = DPMigrateReq(
                    rid=rid,
                    src_dp_rank=src_rank,
                    dst_dp_rank=dst_rank,
                )
                logger.info(
                    f"Sending DPMigrateReq: rid={rid} "
                    f"src={src_rank} dst={dst_rank}"
                )
                self.send_to_tokenizer.send_pyobj(cmd)

    def update_active_ranks(self, ranks: ActiveRanksOutput):
        self.status = ranks.status

    def dispatching_with_trace(self, req: Req):
        if self.server_args.enable_trace:
            trace_set_proc_propagate_context(req.rid, req.trace_context)
            trace_slice_start(RequestStage.DC_DISPATCH, req.rid)
            req.trace_context = trace_get_proc_propagate_context(req.rid)

        self.dispatching(req)

        if self.server_args.enable_trace:
            trace_slice_end(RequestStage.DC_DISPATCH, req.rid, thread_finish_flag=True)

    def init_dispatcher(self):
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (TokenizedGenerateReqInput, self.dispatching_with_trace),
                (TokenizedEmbeddingReqInput, self.dispatching_with_trace),
                (BlockReqInput, self.send_to_all_workers),
                (WatchLoadUpdateReq, self.handle_load_update_req),
                (ActiveRanksOutput, self.update_active_ranks),
            ]
        )
        self._request_dispatcher.add_fallback_fn(self.send_control_message)

    def launch_dp_schedulers(self, server_args, port_args):
        base_gpu_id = 0

        threads = []
        sockets = []
        ready_events = []
        for dp_rank in range(server_args.dp_size):
            tmp_port_args = PortArgs.init_new(server_args)
            tmp_port_args.tokenizer_ipc_name = port_args.tokenizer_ipc_name
            tmp_port_args.detokenizer_ipc_name = port_args.detokenizer_ipc_name

            # This port is checked free in PortArgs.init_new.
            # We hold it first so that the next dp worker gets a different port
            sockets.append(bind_port(tmp_port_args.nccl_port))

            ready_event = threading.Event()
            ready_events.append(ready_event)

            # Create a thread for each worker
            thread = threading.Thread(
                target=self.launch_tensor_parallel_group_thread,
                args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
            )
            threads.append(thread)
            base_gpu_id += (
                server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
            )

            if server_args.node_rank == 0:
                self.workers[dp_rank] = get_zmq_socket(
                    self.context,
                    zmq.PUSH,
                    tmp_port_args.scheduler_input_ipc_name,
                    True,
                )

        # Free all sockets before starting the threads to launch TP workers
        for sock in sockets:
            sock.close()

        # Start all threads
        for thread in threads:
            thread.start()
        for event in ready_events:
            event.wait()

    def launch_tensor_parallel_group_thread(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: int,
        ready_event: threading.Event,
    ):
        self.launch_tensor_parallel_group(server_args, port_args, base_gpu_id, dp_rank)
        ready_event.set()

        # This thread cannot be closed because otherwise the `kill_itself_when_parent_died`
        # function in scheduler.py will kill the scheduler.
        while True:
            time.sleep(30 * 24 * 3600)

    def _broadcast_worker_ports(
        self, server_args: ServerArgs, worker_ports: Optional[List[int]] = None
    ) -> List[int]:
        """Broadcast worker ports from node 0 to all other nodes.

        Node 0 acts as the server, waiting for all other nodes to connect and
        sending them the pre-allocated worker ports. Other nodes act as clients,
        connecting to node 0 to receive their copy of the worker ports.

        Args:
            server_args: Server arguments containing node configuration.
            worker_ports: Pre-allocated worker ports to broadcast.

        Returns:
            List of worker ports (same on all nodes after broadcast).
        """
        # Determine the endpoint for inter-node communication
        if server_args.dist_init_addr is None:
            endpoint = f"tcp://127.0.0.1:{server_args.port + DP_ATTENTION_HANDSHAKE_PORT_DELTA}"
        elif server_args.dist_init_addr.startswith("["):  # ipv6 address
            port, host = configure_ipv6(server_args.dist_init_addr)
            endpoint = f"tcp://{host}:{int(port) + DP_ATTENTION_HANDSHAKE_PORT_DELTA}"
        else:
            host, port = server_args.dist_init_addr.split(":")
            endpoint = f"tcp://{host}:{int(port) + DP_ATTENTION_HANDSHAKE_PORT_DELTA}"

        if server_args.node_rank == 0:
            # Node 0: Broadcast worker ports to all other nodes
            return self._broadcast_ports_as_server(
                endpoint, server_args.nnodes - 1, worker_ports
            )
        else:
            # Other nodes: Receive worker ports from node 0
            return self._receive_ports_as_client(endpoint, server_args.node_rank)

    def _broadcast_ports_as_server(
        self, endpoint: str, expected_clients: int, worker_ports: List[int]
    ) -> List[int]:
        """Broadcast worker ports to all client nodes."""
        logger.debug(f"Broadcasting worker ports to {expected_clients} client nodes")
        logger.debug(f"Worker ports: {worker_ports}")

        rep_socket = get_zmq_socket(self.context, zmq.REP, endpoint, True)

        try:
            connected_clients = 0
            while connected_clients < expected_clients:
                # Wait for client handshake
                client_rank = rep_socket.recv().decode()
                logger.debug(f"Received handshake from node {client_rank}")

                # Send worker ports to client
                rep_socket.send_pyobj(worker_ports)
                connected_clients += 1
                logger.debug(
                    f"Sent worker ports to {connected_clients}/{expected_clients} nodes"
                )

            logger.debug("Worker port broadcast completed")
            return worker_ports
        finally:
            rep_socket.close()

    def _receive_ports_as_client(self, endpoint: str, node_rank: int) -> List[int]:
        """Receive worker ports from the server node."""
        logger.debug(f"Connecting to node 0 to receive worker ports")

        req_socket = get_zmq_socket(self.context, zmq.REQ, endpoint, False)
        req_socket.setsockopt(zmq.RCVTIMEO, 600 * 1000)  # 10 minute timeout
        req_socket.setsockopt(zmq.SNDTIMEO, 600 * 1000)

        try:
            # Send handshake with our node rank
            req_socket.send(str(node_rank).encode())

            # Receive worker ports
            worker_ports = req_socket.recv_pyobj()
            logger.debug(f"Received {len(worker_ports)} worker ports from node 0")
            return worker_ports
        except zmq.Again:
            logger.error("Timeout waiting for worker ports from node 0")
            raise RuntimeError(
                "Failed to receive worker ports from node 0 within timeout"
            )
        finally:
            req_socket.close()

    def launch_dp_attention_schedulers(
        self, server_args: ServerArgs, port_args: PortArgs
    ):
        # Pre-allocate worker ports on node 0 to avoid conflicts
        worker_ports = []
        if server_args.node_rank == 0:
            for dp_rank in range(server_args.dp_size):
                port_and_socket = get_zmq_socket(self.context, zmq.PUSH)
                worker_ports.append(port_and_socket[0])
                self.workers[dp_rank] = port_and_socket[1]
                logger.debug(f"Assigned port {port_and_socket[0]} to worker {dp_rank}")

        broadcasted_ports = self._broadcast_worker_ports(
            server_args, worker_ports if worker_ports else None
        )
        self.launch_tensor_parallel_group(
            server_args, port_args, 0, None, broadcasted_ports
        )

    def launch_tensor_parallel_group(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        base_gpu_id: int,
        dp_rank: Optional[int],
        worker_ports: Optional[List[int]] = None,
    ):
        if not server_args.enable_dp_attention:
            logger.info(f"Launch DP{dp_rank} starting at GPU #{base_gpu_id}.")

        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )

        scheduler_pipe_readers = []

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        nnodes_per_pp_rank = max(server_args.nnodes // server_args.pp_size, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank),
            pp_size_per_node * (server_args.node_rank // nnodes_per_pp_rank + 1),
        )

        nnodes_per_tp_group = nnodes_per_pp_rank
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                rank_port_args = port_args

                if server_args.enable_dp_attention:
                    # dp attention has different sharding logic
                    _, _, dp_rank = compute_dp_attention_world_info(
                        server_args.enable_dp_attention,
                        tp_rank,
                        server_args.tp_size,
                        server_args.dp_size,
                    )
                    # compute zmq ports for this dp rank
                    rank_port_args = PortArgs.init_new(
                        server_args, dp_rank, worker_ports
                    )
                    # Data parallelism reuses the tensor parallelism group,
                    # so all dp ranks should use the same nccl port.
                    rank_port_args.nccl_port = port_args.nccl_port

                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
                with self.env_lock, maybe_reindex_device_id(gpu_id) as gpu_id:
                    proc = mp.Process(
                        target=self.run_scheduler_process_func,
                        args=(
                            server_args,
                            rank_port_args,
                            gpu_id,
                            tp_rank,
                            moe_ep_rank,
                            pp_rank,
                            dp_rank,
                            writer,
                        ),
                    )
                    with memory_saver_adapter.configure_subprocess(), numa_utils.configure_subprocess(
                        server_args, gpu_id
                    ):
                        proc.start()
                self.scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)

        # Wait for model to finish loading
        scheduler_info = []
        for i in range(len(scheduler_pipe_readers)):
            scheduler_info.append(scheduler_pipe_readers[i].recv())

        self.max_total_num_tokens = scheduler_info[0]["max_total_num_tokens"]
        self.max_req_input_len = scheduler_info[0]["max_req_input_len"]

    def maybe_external_dp_rank_routing(self, req: Req):
        if req.data_parallel_rank is not None:
            logger.debug(f"Direct routing to DP rank {req.data_parallel_rank}")
            self.workers[req.data_parallel_rank].send_pyobj(req)
            return True
        return False

    def round_robin_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        while True:
            if self.status[self.round_robin_counter]:
                logger.debug(f"Choose worker {self.round_robin_counter}")
                self.workers[self.round_robin_counter].send_pyobj(req)
                self.round_robin_counter = (self.round_robin_counter + 1) % len(
                    self.workers
                )
                break
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

    def follow_bootstrap_room_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return

        # Set default bootstrap_room if in FAKE auto mode and room is None
        if (
            req.bootstrap_room is None
            and self.server_args.disaggregation_decode_enable_fake_auto
        ):
            req.bootstrap_room = self.round_robin_counter
            self.round_robin_counter = (self.round_robin_counter + 1) % len(
                self.workers
            )

        assert req.bootstrap_room is not None, (
            "req.bootstrap_room should not be None. Do not send requests directly to "
            "prefill or decode instances; send to the router instead."
        )
        target_rank = req.bootstrap_room % len(self.workers)
        self.workers[target_rank].send_pyobj(req)

    def total_requests_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.workers[target_worker].send_pyobj(req)

    def total_tokens_scheduler(self, req: Req):
        if self.maybe_external_dp_rank_routing(req):
            return
        target_worker = self.dp_budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS)
        self.workers[target_worker].send_pyobj(req)

    def event_loop(self):
        while True:
            while True:
                self.soft_watchdog.feed()
                try:
                    recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
                except zmq.ZMQError:
                    break
                self._request_dispatcher(recv_req)


def run_data_parallel_controller_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
    run_scheduler_process_func: Callable = run_scheduler_process,
):
    setproctitle.setproctitle("sglang::data_parallel_controller")
    faulthandler.enable()
    kill_itself_when_parent_died()
    parent_process = psutil.Process().parent()

    configure_logger(server_args)
    if server_args.enable_trace:
        process_tracing_init(server_args.otlp_traces_endpoint, "sglang")
        thread_label = "DP Controller"
        if server_args.disaggregation_mode == "prefill":
            thread_label = "Prefill DP Controller"
        elif server_args.disaggregation_mode == "decode":
            thread_label = "Decode DP Controller"
        trace_set_thread_info(thread_label)

    try:
        controller = DataParallelController(
            server_args, port_args, run_scheduler_process_func
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": controller.max_total_num_tokens,
                "max_req_input_len": controller.max_req_input_len,
            }
        )
        if server_args.node_rank == 0:
            controller.event_loop()
        for proc in controller.scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
    except Exception:
        traceback = get_exception_traceback()
        logger.error(f"DataParallelController hit an exception: {traceback}")
        parent_process.send_signal(signal.SIGQUIT)
