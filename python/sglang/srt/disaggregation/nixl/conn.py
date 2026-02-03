from __future__ import annotations

import dataclasses
import logging
import struct
import threading
import time
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Set

import numpy as np
import numpy.typing as npt
import requests
import torch

from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.common.conn import (
    CommonKVBootstrapServer,
    CommonKVManager,
    CommonKVReceiver,
    CommonKVSender,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous
from sglang.srt.disaggregation.nixl.pinned_buffer_pool import PinnedBufferPool
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Default staging buffer size for Triton KV transfer (256MB)
DEFAULT_TRITON_STAGING_BUFFER_SIZE_MB = 256.0


def _import_triton_kv_transfer():
    """Lazily import Triton KV transfer functions to avoid import errors when not used."""
    try:
        from sglang.srt.layers.attention.triton_ops.kv_transfer import (
            gather_kv_to_pinned_all_layers,
            scatter_kv_with_staging_all_layers,
        )

        return gather_kv_to_pinned_all_layers, scatter_kv_with_staging_all_layers
    except ImportError as e:
        logger.warning(f"[TRITON-KV] Failed to import Triton KV transfer: {e}")
        return None, None

GUARD = "NixlMsgGuard".encode("ascii")
NIXL_WAIT_LOG_INTERVAL_S = envs.SGLANG_NIXL_WAIT_LOG_INTERVAL_S.get()


@dataclasses.dataclass
class TransferInfo:
    """Contains indices for a transfer, sent by KVReceiver. Received by prefill bootstrap thread."""

    room: int
    endpoint: str
    dst_port: int
    agent_name: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    required_dst_info_num: int

    def is_dummy(self):
        return self.dst_kv_indices.size == 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            dst_kv_indices=np.frombuffer(msg[4], dtype=np.int32),
            dst_aux_index=int(msg[5].decode("ascii")),
            required_dst_info_num=int(msg[6].decode("ascii")),
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    """Contains base pointers and other info which only needs to be sent once by KVReceiver. Received by prefill bootstrap thread."""

    room: str
    endpoint: str
    dst_port: int
    agent_name: str
    agent_metadata: bytes
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    gpu_id: int
    decode_tp_size: int
    decode_tp_rank: int
    dst_kv_item_len: int
    # For Triton KV transfer: pinned CPU buffer address and size
    dst_pinned_ptr: int = 0
    dst_pinned_size: int = 0

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        # Handle backwards compatibility - older senders may not include pinned buffer info
        dst_pinned_ptr = 0
        dst_pinned_size = 0
        if len(msg) > 11:
            dst_pinned_ptr = int(msg[11].decode("ascii"))
        if len(msg) > 12:
            dst_pinned_size = int(msg[12].decode("ascii"))
        
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            agent_name=msg[3].decode("ascii"),
            agent_metadata=msg[4],
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[6]) // 8}Q", msg[6])),
            gpu_id=int(msg[7].decode("ascii")),
            decode_tp_size=int(msg[8].decode("ascii")),
            decode_tp_rank=int(msg[9].decode("ascii")),
            dst_kv_item_len=int(msg[10].decode("ascii")),
            dst_pinned_ptr=dst_pinned_ptr,
            dst_pinned_size=dst_pinned_size,
        )


@dataclasses.dataclass
class TransferStatus:
    """Used by KV Receiver to know when a transfer is done."""

    # KV chunks received per sender: {sender_key: set of chunk_ids}
    received_kvs_per_sender: Dict[str, Set[int]] = dataclasses.field(
        default_factory=lambda: defaultdict(set)
    )
    # Expected chunk count per sender (set when is_last=True): {sender_key: expected_count}
    expected_kvs_per_sender: Dict[str, int] = dataclasses.field(default_factory=dict)
    # Number of senders expected to send data.
    num_senders_expected: Optional[int] = None
    # Whether aux data has been received.
    received_aux: bool = False
    # Mark as failed
    is_failure: bool = False

    def is_done(self):
        if self.is_failure:
            return True
        if self.num_senders_expected is None or not self.received_aux:
            return False
        # All senders must have reported their expected count
        if len(self.expected_kvs_per_sender) < self.num_senders_expected:
            return False
        # Each sender must have received all expected chunks
        for sender_key, expected in self.expected_kvs_per_sender.items():
            if len(self.received_kvs_per_sender[sender_key]) != expected:
                return False
        return True

    def is_failed(self):
        return self.is_failure


class NixlKVManager(CommonKVManager):
    def __init__(
        self,
        args: KVArgs,
        disaggregation_mode: DisaggregationMode,
        server_args: ServerArgs,
        is_mla_backend: Optional[bool] = False,
        metrics_collector=None,
    ):
        super().__init__(args, disaggregation_mode, server_args, is_mla_backend, metrics_collector)
        try:
            from nixl._api import nixl_agent
        except ImportError as e:
            raise ImportError(
                "Please install NIXL by following the instructions at "
                "https://github.com/ai-dynamo/nixl/blob/main/README.md "
                "to run SGLang with NixlTransferEngine."
            ) from e
        self.agent = nixl_agent(str(uuid.uuid4()))

        # Store KV transfer method configuration
        self.kv_transfer_method = getattr(server_args, "kv_transfer_method", "legacy")
        self.triton_staging_buffer: Optional[torch.Tensor] = None
        self._pinned_pool: Optional[PinnedBufferPool] = None
        self.triton_pinned_descs = None
        # Store server_args for pinned buffer configuration
        self._server_args = server_args

        # Initialize Triton transfer infrastructure if enabled
        if self.kv_transfer_method == "triton":
            self._init_triton_transfer_buffers()
            logger.debug(
                f"[TRITON-KV] Triton KV transfer enabled, "
                f"staging_buffer={DEFAULT_TRITON_STAGING_BUFFER_SIZE_MB}MB"
            )

        self.register_buffer_to_engine()

        # Receiver infrastructure - always initialize for both modes
        # (decode uses it for receiving, prefill doesn't but it's harmless)
        self.transfer_statuses: Dict[int, TransferStatus] = defaultdict(TransferStatus)
        self.heartbeat_failures: Dict[str, int] = {}
        self.session_pool = defaultdict(requests.Session)
        self.session_pool_lock = threading.Lock()
        self.addr_to_rooms_tracker = defaultdict(set)

        # Heartbeat/timeout settings
        self.heartbeat_interval = max(
            envs.SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL.get(), 2.0
        )
        self.max_failures = max(
            envs.SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE.get(), 1
        )
        self.waiting_timeout = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()

        # Start both threads - enables any node to act as sender or receiver
        self._start_bootstrap_thread()
        self._start_heartbeat_checker_thread()

    def _start_heartbeat_checker_thread(self):
        """
        Start the heartbeat checker thread for Decode worker.
        TODO (smor): unite nixl heartbeat checker with mooncake's.
        """

        def heartbeat_checker():
            while True:
                time.sleep(self.heartbeat_interval)
                with self.connection_lock:
                    addresses = list(self.prefill_dp_size_table.keys())

                for bootstrap_addr in addresses:
                    session = None
                    try:
                        with self.session_pool_lock:
                            session = self.session_pool[bootstrap_addr]
                        response = session.get(
                            f"http://{bootstrap_addr}/health",
                            timeout=(2, 3),
                            headers={"Connection": "keep-alive"},
                        )
                        if response.status_code == 200:
                            self.heartbeat_failures[bootstrap_addr] = 0

                        else:
                            logger.info(
                                f"Attempting to reconnect to {bootstrap_addr}..."
                            )
                            self.heartbeat_failures[bootstrap_addr] = (
                                self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                            )
                            with self.session_pool_lock:
                                if bootstrap_addr in self.session_pool:
                                    del self.session_pool[bootstrap_addr]
                    except Exception:
                        logger.info(f"Attempting to reconnect to {bootstrap_addr}...")
                        self.heartbeat_failures[bootstrap_addr] = (
                            self.heartbeat_failures.get(bootstrap_addr, 0) + 1
                        )

                    if (
                        self.heartbeat_failures.get(bootstrap_addr, 0)
                        >= self.max_failures
                    ):
                        self._handle_node_failure(bootstrap_addr)
                        with self.session_pool_lock:
                            if bootstrap_addr in self.session_pool:
                                del self.session_pool[bootstrap_addr]

        threading.Thread(target=heartbeat_checker, daemon=True).start()

    def _handle_node_failure(self, failed_bootstrap_addr):
        """Handle failure of a prefill node."""
        with self.connection_lock:
            keys_to_remove = [
                k for k in self.connection_pool if k.startswith(failed_bootstrap_addr)
            ]
            for k in keys_to_remove:
                del self.connection_pool[k]
            if failed_bootstrap_addr in self.prefill_attn_tp_size_table:
                del self.prefill_attn_tp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_dp_size_table:
                del self.prefill_dp_size_table[failed_bootstrap_addr]
            if failed_bootstrap_addr in self.prefill_pp_size_table:
                del self.prefill_pp_size_table[failed_bootstrap_addr]

            possible_affected_rooms = self.addr_to_rooms_tracker.get(
                failed_bootstrap_addr, []
            )
            if failed_bootstrap_addr in self.addr_to_rooms_tracker:
                del self.addr_to_rooms_tracker[failed_bootstrap_addr]

        # Mark all pending transfers associated with the failed node as failed
        affected_rooms = []
        for room in possible_affected_rooms:
            if (
                room in self.transfer_statuses
                and not self.transfer_statuses[room].is_done()
            ):
                # Mark the transfer as failed
                self.transfer_statuses[room].is_failure = True
                affected_rooms.append(room)

        logger.error(
            f"Lost connection with prefill instance (bootstrap_addr: {failed_bootstrap_addr}), "
            f"{len(affected_rooms)} transfers affected"
        )
        for room in possible_affected_rooms:
            logger.error(f"Let room {room} be failed due to prefill down")
            self.update_status(room, KVPoll.Failed)

    def check_status(self, bootstrap_room: int):
        return self.request_status[bootstrap_room]

    def update_status(self, bootstrap_room: int, status: KVPoll):
        if bootstrap_room not in self.request_status:
            self.request_status[bootstrap_room] = status
        else:
            # NOTE: status is only allowed to be incremented unless it is KVPoll.Failed
            if status == KVPoll.Failed:
                self.request_status[bootstrap_room] = KVPoll.Failed
            else:
                self.request_status[bootstrap_room] = max(
                    self.request_status[bootstrap_room], status
                )

    def record_failure(self, bootstrap_room: int, failure_reason: str):
        pass

    def register_buffer_to_engine(self):
        kv_addrs = []
        for kv_data_ptr, kv_data_len in zip(
            self.kv_args.kv_data_ptrs, self.kv_args.kv_data_lens
        ):
            kv_addrs.append((kv_data_ptr, kv_data_len, self.kv_args.gpu_id, ""))
        self.kv_descs = self.agent.register_memory(kv_addrs, "VRAM")
        logger.debug(f"Register kv tensors, len(kv_addr)= {len(kv_addrs)}")
        if not self.kv_descs:
            raise Exception("NIXL memory registration failed for kv tensors")
        aux_addrs = []
        for aux_data_ptr, aux_data_len in zip(
            self.kv_args.aux_data_ptrs, self.kv_args.aux_data_lens
        ):
            aux_addrs.append((aux_data_ptr, aux_data_len, 0, ""))
        self.aux_descs = self.agent.register_memory(aux_addrs, "DRAM")
        logger.debug(f"Register aux tensors, len(aux_addrs)= {len(aux_addrs)}")
        if not self.aux_descs:
            raise Exception("NIXL memory registration failed for aux tensors")

        # Register shared pinned buffer pool with NIXL if enabled
        if self.kv_transfer_method == "triton" and self._pinned_pool is not None:
            self.triton_pinned_descs = self._pinned_pool.register_with_nixl(self.agent)

    def _init_triton_transfer_buffers(self):
        """Initialize GPU staging buffer and shared pinned buffer pool for Triton KV transfer."""
        # Get dtype from KV cache buffers (supports fp8, fp16, bf16)
        k_buffers = self.kv_args.k_buffers
        if k_buffers is not None and len(k_buffers) > 0:
            kv_dtype = k_buffers[0].dtype
            kv_elem_bytes = k_buffers[0].element_size()
        else:
            # Fallback to bfloat16 if k_buffers not available yet
            kv_dtype = torch.bfloat16
            kv_elem_bytes = 2
            logger.warning(
                "[TRITON-KV] k_buffers not available, falling back to bfloat16. "
                "This may cause issues if KV cache uses a different dtype (e.g., fp8)."
            )

        # Allocate GPU staging buffer (fixed size, 256MB by default)
        staging_size_bytes = int(DEFAULT_TRITON_STAGING_BUFFER_SIZE_MB * 1e6)
        staging_elements = staging_size_bytes // kv_elem_bytes
        self.triton_staging_buffer = torch.empty(
            staging_elements, dtype=kv_dtype, device=f"cuda:{self.kv_args.gpu_id}"
        )

        # Get or create shared pinned buffer pool for this GPU
        # Uses configurable size from --pinned-buffer-max-gb (default 64GB)
        pinned_size_bytes = int(
            getattr(self._server_args, "pinned_buffer_max_gb", 64.0) * 1e9
        )
        self._pinned_pool = PinnedBufferPool.get_or_create(
            gpu_id=self.kv_args.gpu_id,
            dtype=kv_dtype,
            total_size_bytes=pinned_size_bytes,
        )

        logger.debug(
            f"[TRITON-KV] Initialized transfer buffers: "
            f"staging={self.triton_staging_buffer.nbytes / 1e6:.2f}MB (GPU), "
            f"shared_pinned_pool={pinned_size_bytes / 1e9:.2f}GB (CPU)"
        )

    def _add_remote_peer(self, decode_kv_args: KVArgsRegisterInfo):
        agent_name = decode_kv_args.agent_name
        if agent_name in self.decode_kv_args_table:
            logger.info(f"Peer {agent_name} was already registered, ignoring.")
            return
        self.decode_kv_args_table[agent_name] = decode_kv_args
        self.agent.add_remote_agent(decode_kv_args.agent_metadata)

    def send_kvcache(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
    ):
        # group by indices
        prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
            prefill_kv_indices, dst_kv_indices
        )

        logger.debug(f"sending kvcache to {peer_name} with notif {notif}")
        # Make descs
        if self.is_mla_backend:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            )
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]
        else:
            src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
                self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
            )

            layers_params = [
                (
                    src_k_ptrs[layer_id],
                    dst_k_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ] + [
                (
                    src_v_ptrs[layer_id],
                    dst_v_ptrs[layer_id],
                    self.kv_args.kv_item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
            ]

        src_addrs = []
        dst_addrs = []
        for src_ptr, dst_ptr, item_len in layers_params:
            for prefill_index, decode_index in zip(prefill_kv_blocks, dst_kv_blocks):
                src_addr = src_ptr + int(prefill_index[0]) * item_len
                dst_addr = dst_ptr + int(decode_index[0]) * item_len
                length = item_len * len(prefill_index)
                src_addrs.append((src_addr, length, self.kv_args.gpu_id))
                dst_addrs.append((dst_addr, length, dst_gpu_id))

        logger.debug(
            f"len(src_addrs): before group: {len(prefill_kv_indices)}, after group: {len(src_addrs)}"
        )
        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def send_kvcache_slice(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_kv_ptrs: list[int],
        dst_kv_indices: npt.NDArray[np.int32],
        dst_gpu_id: int,
        notif: str,
        prefill_tp_size: int,
        decode_tp_size: int,
        decode_tp_rank: int,
        dst_kv_item_len: int,
    ):
        # Get configuration from kv_args
        local_tp_rank_in_group = self.kv_args.engine_rank % prefill_tp_size
        dst_tp_rank_in_group = decode_tp_rank % decode_tp_size
        num_kv_heads = self.kv_args.kv_head_num

        # Calculate head distribution
        src_heads_per_rank = num_kv_heads
        dst_heads_per_rank = num_kv_heads * prefill_tp_size // decode_tp_size

        src_kv_item_len = self.kv_args.kv_item_lens[0]
        page_size = self.kv_args.page_size

        bytes_per_head_slice_to_send = (
            dst_kv_item_len // page_size // dst_heads_per_rank
        )

        # Determine which heads to send
        if prefill_tp_size > decode_tp_size:
            # Multiple prefill ranks to one decode rank
            src_head_start_offset = 0
            num_heads_to_send = src_heads_per_rank
            dst_head_start_offset = local_tp_rank_in_group * src_heads_per_rank
        else:
            # Send KVCache from 1 prefill instance to multiple decode instances
            src_head_start_offset = (
                dst_tp_rank_in_group * dst_heads_per_rank
            ) % src_heads_per_rank
            num_heads_to_send = dst_heads_per_rank
            dst_head_start_offset = 0

        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(self.kv_args.kv_data_ptrs, dst_kv_ptrs)
        )
        # Create transfer descriptors
        src_addrs = []
        dst_addrs = []

        bytes_per_token_on_prefill = src_kv_item_len // page_size
        bytes_per_token_on_decode = dst_kv_item_len // page_size

        # Calculate precise byte offset and length for the sub-slice within the token
        src_head_slice_offset = src_head_start_offset * bytes_per_head_slice_to_send
        dst_head_slice_offset = dst_head_start_offset * bytes_per_head_slice_to_send
        heads_bytes_per_token_to_send = num_heads_to_send * bytes_per_head_slice_to_send

        src_dst_ptr_pairs = [
            (
                src_k_ptrs[layer_id],
                dst_k_ptrs[layer_id],
            )
            for layer_id in range(layers_current_pp_stage)
        ] + [
            (
                src_v_ptrs[layer_id],
                dst_v_ptrs[layer_id],
            )
            for layer_id in range(layers_current_pp_stage)
        ]

        src_addrs = []
        dst_addrs = []

        # Calculate strides for a single token slot
        bytes_per_token_on_prefill = src_kv_item_len // page_size
        bytes_per_token_on_decode = dst_kv_item_len // page_size

        for src_ptr, dst_ptr in src_dst_ptr_pairs:
            for i in range(len(prefill_kv_indices)):
                prefill_page_idx = int(prefill_kv_indices[i])
                decode_page_idx = int(dst_kv_indices[i])

                # Get the starting addresses for the current src and dst pages
                src_page_start_addr = src_ptr + prefill_page_idx * src_kv_item_len
                dst_page_start_addr = dst_ptr + decode_page_idx * dst_kv_item_len

                # Iterate through each valid token slot within the current page
                for token_slot_in_page in range(page_size):
                    # Calculate the start address of the current token slot
                    src_token_slot_start_addr = (
                        src_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_prefill
                    )
                    dst_token_slot_start_addr = (
                        dst_page_start_addr
                        + token_slot_in_page * bytes_per_token_on_decode
                    )

                    # Calculate final src and dst addresses by applying head-slice offsets
                    src_slice_addr = src_token_slot_start_addr + src_head_slice_offset
                    dst_slice_addr = dst_token_slot_start_addr + dst_head_slice_offset

                    src_addrs.append(
                        (
                            src_slice_addr,
                            heads_bytes_per_token_to_send,
                            self.kv_args.gpu_id,
                        )
                    )
                    dst_addrs.append(
                        (dst_slice_addr, heads_bytes_per_token_to_send, dst_gpu_id)
                    )

        # Use NIXL agent for transfer
        src_descs = self.agent.get_xfer_descs(src_addrs, "VRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "VRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not xfer_handle:
            raise Exception("Failed to create sliced KV transfer")

        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("Failed to post sliced KV transfer")

        return xfer_handle

    def _expand_pages_to_slots(
        self,
        page_indices: npt.NDArray[np.int32],
        page_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Expand page indices to slot indices (each page has page_size slots)."""
        pages = torch.from_numpy(page_indices).to(device, dtype=torch.int64)
        # For each page, generate slot indices: page_idx * page_size + [0, 1, ..., page_size-1]
        offsets = torch.arange(page_size, device=device, dtype=torch.int64)
        return (pages.unsqueeze(1) * page_size + offsets).flatten()

    def send_kvcache_triton(
        self,
        peer_name: str,
        prefill_kv_indices: npt.NDArray[np.int32],
        dst_pinned_ptr: int,
        dst_pinned_size: int,
        notif: str,
        head_start: int = 0,
        num_heads_to_send: int = None,
        dst_head_offset: int = 0,
    ):
        """
        Send KV cache using Triton gather kernel + single NIXL transfer.

        This method:
        1. Allocates a region from the shared pinned buffer pool
        2. Uses gather_kv_to_pinned_all_layers to collect scattered KV data into the region
           (single kernel launch for all layers)
        3. Transfers the contiguous pinned buffer to destination pinned buffer via NIXL
        4. Destination will use scatter_kv to distribute to its KV cache

        This reduces NIXL descriptor count from O(tokens × layers) to O(1).

        Args:
            peer_name: NIXL peer agent name
            prefill_kv_indices: Indices of KV slots to transfer
            dst_pinned_ptr: Destination pinned buffer pointer
            dst_pinned_size: Destination pinned buffer size in bytes
            notif: Notification message for transfer completion
            head_start: First head index to gather (for head slicing)
            num_heads_to_send: Number of heads to gather
        """
        gather_kv_all_layers, _ = _import_triton_kv_transfer()
        if gather_kv_all_layers is None:
            raise RuntimeError(
                "[TRITON-KV] Triton KV transfer not available. "
                "Make sure triton is installed."
            )

        # Validate we have tensor buffers available
        if self.kv_args.k_buffers is None or self.kv_args.v_buffers is None:
            raise RuntimeError(
                "[TRITON-KV] k_buffers and v_buffers must be set in KVArgs "
                "when using Triton KV transfer."
            )

        # Validate we have shared pinned pool
        if self._pinned_pool is None:
            raise RuntimeError(
                "[TRITON-KV] Pinned buffer pool not initialized. "
                "Ensure Triton transfer is enabled."
            )

        # Get tensor buffers (these should be the same as kv_data_ptrs)
        k_buffers = self.kv_args.k_buffers
        v_buffers = self.kv_args.v_buffers
        num_layers = len(k_buffers)
        num_heads = k_buffers[0].shape[1]
        head_dim = k_buffers[0].shape[2]
        device = k_buffers[0].device

        if num_heads_to_send is None:
            num_heads_to_send = num_heads - head_start

        # Convert page indices to slot indices (each page has page_size slots)
        page_size = self.kv_args.page_size
        slot_indices_tensor = self._expand_pages_to_slots(
            prefill_kv_indices, page_size, device
        ).to(torch.int32)
        num_tokens = len(slot_indices_tensor)

        # Calculate transfer size AFTER page-to-slot expansion
        bytes_per_element = k_buffers[0].element_size()
        transfer_elements = num_layers * 2 * num_tokens * num_heads_to_send * head_dim
        transfer_bytes = transfer_elements * bytes_per_element

        # Allocate region from shared pinned buffer pool
        src_offset, buffer_region = self._pinned_pool.allocate(transfer_bytes)

        logger.debug(
            f"[TRITON-KV] send_kvcache_triton: {num_tokens} tokens, {num_layers} layers, "
            f"heads [{head_start}:{head_start + num_heads_to_send}], "
            f"transfer_size={transfer_bytes / 1e6:.2f}MB, pool_offset={src_offset}"
        )

        # Create pointer tensors for single-kernel gather (cached for reuse)
        if not hasattr(self, '_k_data_ptrs') or self._k_data_ptrs is None:
            self._k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in k_buffers], dtype=torch.uint64, device=device
            )
            self._v_data_ptrs = torch.tensor(
                [x.data_ptr() for x in v_buffers], dtype=torch.uint64, device=device
            )
            self._src_slot_stride = k_buffers[0].stride(0)
            self._src_head_stride = k_buffers[0].stride(1)

        # Step 1: Gather KV data to allocated region using single-kernel Triton (device->host)
        t0_gather = time.perf_counter()
        gather_kv_all_layers(
            k_data_ptrs=self._k_data_ptrs,
            v_data_ptrs=self._v_data_ptrs,
            slot_indices=slot_indices_tensor,
            pinned_output=buffer_region,
            head_start=head_start,
            num_heads_to_gather=num_heads_to_send,
            num_layers=num_layers,
            head_dim=head_dim,
            src_slot_stride=self._src_slot_stride,
            src_head_stride=self._src_head_stride,
            kv_elem_bytes=bytes_per_element,
        )
        torch.cuda.synchronize()  # Ensure gather is complete
        t1_gather = time.perf_counter()
        gather_ms = (t1_gather - t0_gather) * 1000
        transfer_mb = transfer_bytes / 1e6
        gather_speed_gb_s = transfer_bytes / gather_ms / 1e6
        logger.info(
            f"[TRITON-KV] Device->Host transfer (gather_kv_all_layers) took {gather_ms:.2f}ms "
            f"({transfer_mb:.2f}MB, {gather_speed_gb_s:.2f} GB/s)"
        )

        # Record metrics if metrics collector is available
        if self.metrics_collector is not None:
            self.metrics_collector._log_histogram(
                self.metrics_collector.kv_triton_gather_latency_ms, gather_ms
            )
            self.metrics_collector._log_histogram(
                self.metrics_collector.kv_triton_gather_speed_gb_s, gather_speed_gb_s
            )
            self.metrics_collector._log_histogram(
                self.metrics_collector.kv_triton_transfer_size_mb, transfer_mb
            )

        # Step 2: Transfer pinned buffer to destination via NIXL (single transfer)
        # Calculate destination offset based on dst_head_offset for mixed-TP transfers
        head_stride_bytes = num_layers * 2 * num_tokens * head_dim * bytes_per_element
        dst_offset = dst_head_offset * head_stride_bytes

        # Debug: Check if transfer would exceed destination buffer
        required_end = dst_offset + transfer_bytes
        logger.info(
            f"[TRITON-KV] Transfer check: dst_pinned_ptr=0x{dst_pinned_ptr:x}, "
            f"dst_pinned_size={dst_pinned_size}, dst_offset={dst_offset}, "
            f"transfer_bytes={transfer_bytes}, required_end={required_end}, "
            f"head_stride_bytes={head_stride_bytes}, dst_head_offset={dst_head_offset}"
        )
        if required_end > dst_pinned_size:
            logger.error(
                f"[TRITON-KV] BUFFER OVERFLOW! Need {required_end} bytes but dst buffer is only {dst_pinned_size} bytes. "
                f"Overflow by {required_end - dst_pinned_size} bytes ({(required_end - dst_pinned_size) / 1e6:.2f} MB)"
            )

        # Use allocated region's data pointer for source address
        src_addrs = [
            (buffer_region.data_ptr(), transfer_bytes, 0)
        ]
        dst_addrs = [
            (dst_pinned_ptr + dst_offset, transfer_bytes, 0)
        ]

        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")

        xfer_handle = self.agent.initialize_xfer(
            "WRITE", src_descs, dst_descs, peer_name, notif.encode("ascii")
        )
        if not xfer_handle:
            # Release allocation on failure
            self._pinned_pool.release(src_offset)
            raise Exception("[TRITON-KV] Failed to create Triton KV transfer")

        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            # Release allocation on failure
            self._pinned_pool.release(src_offset)
            raise Exception("[TRITON-KV] Failed to post Triton KV transfer")

        logger.debug(
            f"[TRITON-KV] Transfer initiated: {transfer_bytes / 1e6:.2f}MB "
            f"({num_tokens} tokens × {num_layers} layers)"
        )

        # Return handle and pool allocation info for later release
        return (xfer_handle, self._pinned_pool, src_offset)

    def _send_kvcache_triton_batched(
        self,
        requests: List[tuple],  # List of (agent_name, dst_pinned_ptr, dst_pinned_size, notif, head_start, num_heads)
        prefill_kv_indices: npt.NDArray[np.int32],
        total_heads: int,
    ):
        """
        Batched KV transfer: ONE gather of ALL heads, then slice buffer for parallel NIXL transfers.

        This eliminates the performance bottleneck of doing N serial gathers when
        sender_tp < receiver_tp (e.g., prefill TP=1 sending to decode TP=8).

        Args:
            requests: List of (agent_name, dst_pinned_ptr, dst_pinned_size, notif, head_start, num_heads)
            prefill_kv_indices: Page indices to transfer
            total_heads: Total number of KV heads on this prefill rank

        Returns:
            Tuple of (handles, pool_allocations)
        """
        gather_kv_all_layers, _ = _import_triton_kv_transfer()
        if gather_kv_all_layers is None:
            raise RuntimeError("[TRITON-KV] Triton KV transfer not available.")

        if self.kv_args.k_buffers is None or self.kv_args.v_buffers is None:
            raise RuntimeError("[TRITON-KV] k_buffers and v_buffers must be set.")

        if self._pinned_pool is None:
            raise RuntimeError("[TRITON-KV] Pinned buffer pool not initialized.")

        # Get tensor buffers
        k_buffers = self.kv_args.k_buffers
        v_buffers = self.kv_args.v_buffers
        num_layers = len(k_buffers)
        head_dim = k_buffers[0].shape[2]
        device = k_buffers[0].device

        # Convert page indices to slot indices
        page_size = self.kv_args.page_size
        slot_indices_tensor = self._expand_pages_to_slots(
            prefill_kv_indices, page_size, device
        ).to(torch.int32)
        num_tokens = len(slot_indices_tensor)

        # Calculate total buffer size for ALL heads
        bytes_per_element = k_buffers[0].element_size()
        total_transfer_bytes = num_layers * 2 * num_tokens * total_heads * head_dim * bytes_per_element

        # Allocate ONE buffer from pool for all heads
        src_offset, buffer_region = self._pinned_pool.allocate(total_transfer_bytes)

        logger.debug(
            f"[TRITON-KV-BATCHED] Batched gather: {len(requests)} destinations, "
            f"{num_tokens} tokens, {total_heads} heads, "
            f"total_size={total_transfer_bytes / 1e6:.2f}MB"
        )

        # Create pointer tensors (cached for reuse)
        if not hasattr(self, '_k_data_ptrs') or self._k_data_ptrs is None:
            self._k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in k_buffers], dtype=torch.uint64, device=device
            )
            self._v_data_ptrs = torch.tensor(
                [x.data_ptr() for x in v_buffers], dtype=torch.uint64, device=device
            )
            self._src_slot_stride = k_buffers[0].stride(0)
            self._src_head_stride = k_buffers[0].stride(1)

        # Step 1: ONE gather of ALL heads
        t0_gather = time.perf_counter()
        gather_kv_all_layers(
            k_data_ptrs=self._k_data_ptrs,
            v_data_ptrs=self._v_data_ptrs,
            slot_indices=slot_indices_tensor,
            pinned_output=buffer_region,
            head_start=0,  # Gather all heads starting from 0
            num_heads_to_gather=total_heads,
            num_layers=num_layers,
            head_dim=head_dim,
            src_slot_stride=self._src_slot_stride,
            src_head_stride=self._src_head_stride,
            kv_elem_bytes=bytes_per_element,
        )
        torch.cuda.synchronize()  # ONE synchronize for all destinations
        t1_gather = time.perf_counter()
        gather_ms = (t1_gather - t0_gather) * 1000
        gather_speed_gb_s = total_transfer_bytes / gather_ms / 1e6

        logger.debug(
            f"[TRITON-KV-BATCHED] Single gather took {gather_ms:.2f}ms "
            f"({total_transfer_bytes / 1e6:.2f}MB, {gather_speed_gb_s:.2f} GB/s) "
            f"for {len(requests)} destinations"
        )

        # Step 2: Calculate head stride for slicing
        # Buffer layout: [head, layer, kv, token, head_dim]
        head_stride_bytes = num_layers * 2 * num_tokens * head_dim * bytes_per_element

        # Step 3: Loop over requests, calculate slice offsets, initiate NIXL transfers
        handles = []
        for agent_name, dst_pinned_ptr, dst_pinned_size, notif, head_start, num_heads in requests:
            # Calculate source slice pointer within our gathered buffer
            src_slice_ptr = buffer_region.data_ptr() + head_start * head_stride_bytes
            slice_bytes = num_heads * head_stride_bytes

            # Destination offset is 0 since each decode rank has its own buffer
            # and expects the data starting from the beginning
            dst_offset = 0

            # Debug check
            if dst_offset + slice_bytes > dst_pinned_size:
                logger.error(
                    f"[TRITON-KV-BATCHED] BUFFER OVERFLOW! "
                    f"dst_offset={dst_offset}, slice_bytes={slice_bytes}, "
                    f"dst_pinned_size={dst_pinned_size}"
                )

            src_addrs = [(src_slice_ptr, slice_bytes, 0)]
            dst_addrs = [(dst_pinned_ptr + dst_offset, slice_bytes, 0)]

            src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
            dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")

            xfer_handle = self.agent.initialize_xfer(
                "WRITE", src_descs, dst_descs, agent_name, notif.encode("ascii")
            )
            if not xfer_handle:
                self._pinned_pool.release(src_offset)
                raise Exception(f"[TRITON-KV-BATCHED] Failed to create transfer to {agent_name}")

            state = self.agent.transfer(xfer_handle)
            if state == "ERR":
                self._pinned_pool.release(src_offset)
                raise Exception(f"[TRITON-KV-BATCHED] Failed to post transfer to {agent_name}")

            handles.append(xfer_handle)
            logger.debug(
                f"[TRITON-KV-BATCHED] Transfer {agent_name}: "
                f"heads=[{head_start}:{head_start + num_heads}], "
                f"slice_bytes={slice_bytes / 1e6:.2f}MB"
            )

        # Return (handles, [(pool, offset)]) - single allocation for all transfers
        return handles, [(self._pinned_pool, src_offset)]

    def scatter_received_kv(
        self,
        kv_indices: npt.NDArray[np.int32],
        head_start: int = 0,
        num_heads_received: int = None,
    ):
        """
        Scatter received KV data from pinned buffer to KV cache.

        This should be called on the receiver side after NIXL transfer completes
        when using Triton KV transfer method.

        Args:
            kv_indices: Indices of KV slots to scatter to
            head_start: First head index to scatter to (for head slicing)
            num_heads_received: Number of heads received
        """
        _, scatter_kv_all_layers = _import_triton_kv_transfer()
        if scatter_kv_all_layers is None:
            raise RuntimeError(
                "[TRITON-KV] Triton KV transfer not available. "
                "Make sure triton is installed."
            )

        # Validate we have tensor buffers available
        if self.kv_args.k_buffers is None or self.kv_args.v_buffers is None:
            raise RuntimeError(
                "[TRITON-KV] k_buffers and v_buffers must be set in KVArgs "
                "when using Triton KV transfer."
            )

        if self._pinned_pool is None:
            raise RuntimeError(
                "[TRITON-KV] Pinned buffer pool not initialized. "
                "Ensure Triton transfer is enabled."
            )

        # Get tensor buffers
        k_buffers = self.kv_args.k_buffers
        v_buffers = self.kv_args.v_buffers
        num_layers = len(k_buffers)
        num_heads = k_buffers[0].shape[1]
        head_dim = k_buffers[0].shape[2]
        device = k_buffers[0].device

        if num_heads_received is None:
            num_heads_received = num_heads - head_start

        # Convert page indices to slot indices (each page has page_size slots)
        page_size = self.kv_args.page_size
        slot_indices_tensor = self._expand_pages_to_slots(
            kv_indices, page_size, device
        ).to(torch.int32)
        num_tokens = len(slot_indices_tensor)

        logger.debug(
            f"[TRITON-KV] scatter_received_kv: {num_tokens} tokens, {num_layers} layers, "
            f"heads [{head_start}:{head_start + num_heads_received}]"
        )

        # Calculate transfer size for bandwidth measurement
        bytes_per_element = k_buffers[0].element_size()
        transfer_elements = num_layers * 2 * num_tokens * num_heads_received * head_dim
        transfer_bytes = transfer_elements * bytes_per_element

        # Create pointer tensors for single-kernel scatter (cached for reuse)
        if not hasattr(self, '_k_data_ptrs') or self._k_data_ptrs is None:
            self._k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in k_buffers], dtype=torch.uint64, device=device
            )
            self._v_data_ptrs = torch.tensor(
                [x.data_ptr() for x in v_buffers], dtype=torch.uint64, device=device
            )
            self._dst_slot_stride = k_buffers[0].stride(0)
            self._dst_head_stride = k_buffers[0].stride(1)

        # Scatter from shared pool's buffer to KV cache using single-kernel (host->device)
        t0_scatter = time.perf_counter()
        scatter_kv_all_layers(
            pinned_input=self._pinned_pool.buffer,
            k_data_ptrs=self._k_data_ptrs,
            v_data_ptrs=self._v_data_ptrs,
            slot_indices=slot_indices_tensor,
            head_start=head_start,
            num_heads_to_scatter=num_heads_received,
            num_layers=num_layers,
            head_dim=head_dim,
            dst_slot_stride=self._dst_slot_stride,
            dst_head_stride=self._dst_head_stride,
            kv_elem_bytes=bytes_per_element,
        )
        torch.cuda.synchronize()  # Ensure scatter is complete
        t1_scatter = time.perf_counter()
        scatter_ms = (t1_scatter - t0_scatter) * 1000
        transfer_mb = transfer_bytes / 1e6
        scatter_speed_gb_s = transfer_bytes / scatter_ms / 1e6
        logger.info(
            f"[TRITON-KV] Host->Device transfer (scatter_kv_all_layers) took {scatter_ms:.2f}ms "
            f"({transfer_mb:.2f}MB, {scatter_speed_gb_s:.2f} GB/s)"
        )

        # Record metrics if metrics collector is available
        if self.metrics_collector is not None:
            self.metrics_collector._log_histogram(
                self.metrics_collector.kv_triton_scatter_latency_ms, scatter_ms
            )
            self.metrics_collector._log_histogram(
                self.metrics_collector.kv_triton_scatter_speed_gb_s, scatter_speed_gb_s
            )
            self.metrics_collector._log_histogram(
                self.metrics_collector.kv_triton_transfer_size_mb, transfer_mb
            )

        logger.debug(
            f"[TRITON-KV] Scatter complete: {num_tokens} tokens × {num_layers} layers"
        )

    def send_aux(
        self,
        peer_name: str,
        prefill_aux_index: int,
        dst_aux_ptrs: list[int],
        dst_aux_index: int,
        notif: str,
    ):
        src_addrs = []
        dst_addrs = []

        prefill_aux_ptrs = self.kv_args.aux_data_ptrs
        prefill_aux_item_lens = self.kv_args.aux_item_lens

        for i, _ in enumerate(dst_aux_ptrs):
            length = prefill_aux_item_lens[i]
            src_addr = prefill_aux_ptrs[i] + length * prefill_aux_index
            dst_addr = dst_aux_ptrs[i] + length * dst_aux_index
            src_addrs.append((src_addr, length, 0))
            dst_addrs.append((dst_addr, length, 0))

        src_descs = self.agent.get_xfer_descs(src_addrs, "DRAM")
        dst_descs = self.agent.get_xfer_descs(dst_addrs, "DRAM")
        # Transfer data
        xfer_handle = self.agent.initialize_xfer(
            "WRITE",
            src_descs,
            dst_descs,
            peer_name,
            notif.encode("ascii"),  # type: ignore
        )
        if not xfer_handle:
            raise Exception("KVSender failed to create transfer")
        state = self.agent.transfer(xfer_handle)
        if state == "ERR":
            raise Exception("KVSender failed to post transfer")
        return xfer_handle

    def add_transfer_request(
        self,
        bootstrap_room: int,
        kv_indices: npt.NDArray[np.int32],
        index_slice: slice,
        is_last: bool,
        chunk_id: int,
        aux_index: Optional[int] = None,
    ):
        """
        Add a transfer request for KV cache data.

        Returns:
            Tuple of (handles, pool_allocations) where:
            - handles: List of NIXL transfer handles
            - pool_allocations: List of (pool, offset) tuples for later release
        """
        # Only require aux_index if we have aux data to send (e.g., for EAGLE speculation)
        has_aux_data = len(self.kv_args.aux_data_ptrs) > 0
        assert not is_last or not has_aux_data or aux_index is not None

        reqs_to_be_processed = self.transfer_infos[bootstrap_room].values()
        handles = []
        pool_allocations = []

        # Filter out dummy requests
        active_reqs = [req for req in reqs_to_be_processed if not req.is_dummy()]

        # Detect batched Triton case: sender_tp < receiver_tp with multiple destinations
        # This is the optimization that avoids N serial gathers + synchronizes
        if active_reqs:
            first_decode_info = self.decode_kv_args_table.get(active_reqs[0].agent_name)
            if first_decode_info:
                prefill_tp_size = self.attn_tp_size
                decode_tp_size = first_decode_info.decode_tp_size

                use_batched = (
                    self.kv_transfer_method == "triton"
                    and prefill_tp_size < decode_tp_size
                    and all(
                        self.decode_kv_args_table[r.agent_name].dst_pinned_ptr != 0
                        for r in active_reqs
                    )
                    and self.kv_args.k_buffers is not None
                    and self.kv_args.v_buffers is not None
                    and not self.is_mla_backend
                )

                # Error if sender_tp < receiver_tp but batched Triton can't be used
                if (
                    self.kv_transfer_method == "triton"
                    and prefill_tp_size < decode_tp_size
                    and not use_batched
                ):
                    reasons = []
                    missing_pinned = [
                        r.agent_name for r in active_reqs
                        if self.decode_kv_args_table[r.agent_name].dst_pinned_ptr == 0
                    ]
                    if missing_pinned:
                        reasons.append(f"receivers missing pinned buffer: {missing_pinned}")
                    if self.kv_args.k_buffers is None:
                        reasons.append("sender k_buffers is None")
                    if self.kv_args.v_buffers is None:
                        reasons.append("sender v_buffers is None")
                    if self.is_mla_backend:
                        reasons.append("MLA backend not supported for mixed TP Triton transfer")
                    raise RuntimeError(
                        f"[TRITON-KV] Mixed TP transfer (sender_tp={prefill_tp_size}, "
                        f"receiver_tp={decode_tp_size}) requires batched Triton transfer, "
                        f"but it could not be enabled. Reasons: {', '.join(reasons)}. "
                        f"Legacy transfer is not supported for sender_tp < receiver_tp."
                    )

                if use_batched:
                    # Collect batch request info
                    num_kv_heads = self.kv_args.kv_head_num
                    total_prefill_heads = num_kv_heads * prefill_tp_size
                    heads_per_decode_rank = total_prefill_heads // decode_tp_size

                    batch_requests = []
                    for req in active_reqs:
                        decode_info = self.decode_kv_args_table[req.agent_name]
                        decode_tp_rank = decode_info.decode_tp_rank % decode_tp_size
                        head_start = decode_tp_rank * heads_per_decode_rank
                        notif = f"{req.room}_kv_{chunk_id}_{int(is_last)}_{self.kv_args.pp_rank}"
                        batch_requests.append((
                            req.agent_name,
                            decode_info.dst_pinned_ptr,
                            decode_info.dst_pinned_size,
                            notif,
                            head_start,
                            heads_per_decode_rank,
                        ))

                    logger.debug(
                        f"[TRITON-KV-BATCHED] Using batched path: room={bootstrap_room}, "
                        f"num_destinations={len(batch_requests)}, "
                        f"prefill_tp={prefill_tp_size}, decode_tp={decode_tp_size}, "
                        f"total_heads={num_kv_heads}"
                    )

                    # Single batched call for all destinations
                    batch_handles, batch_allocs = self._send_kvcache_triton_batched(
                        batch_requests, kv_indices, num_kv_heads
                    )
                    handles.extend(batch_handles)
                    pool_allocations.extend(batch_allocs)

                    # Handle aux data separately (unchanged from original logic)
                    if is_last and self.pp_group.is_last_rank and has_aux_data:
                        for req in active_reqs:
                            assert aux_index is not None
                            decode_info = self.decode_kv_args_table[req.agent_name]
                            aux_xfer_handle = self.send_aux(
                                req.agent_name,
                                aux_index,
                                decode_info.dst_aux_ptrs,
                                req.dst_aux_index,
                                str(req.room) + "_aux",
                            )
                            handles.append(aux_xfer_handle)

                    if is_last:
                        del self.transfer_infos[bootstrap_room]

                    return handles, pool_allocations

        for req in reqs_to_be_processed:
            assert bootstrap_room == req.room
            if req.is_dummy():
                continue

            logger.debug(
                f"add_transfer_request: room={bootstrap_room}, "
                f"index_slice={index_slice}, "
                f"kv_indices_len={len(kv_indices)}, "
                f"dst_kv_indices_total_len={len(req.dst_kv_indices)}, "
                f"chunk_id={chunk_id}, is_last={is_last}"
            )
            chunked_dst_kv_indice = req.dst_kv_indices[index_slice]
            if len(chunked_dst_kv_indice) != len(kv_indices):
                logger.error(
                    f"KV indices mismatch! chunked_dst_kv_indice_len={len(chunked_dst_kv_indice)}, "
                    f"kv_indices_len={len(kv_indices)}, "
                    f"dst_kv_indices={req.dst_kv_indices.tolist()}, "
                    f"index_slice={index_slice}"
                )
            assert len(chunked_dst_kv_indice) == len(kv_indices)
            assert req.agent_name in self.decode_kv_args_table

            notif = f"{req.room}_kv_{chunk_id}_{int(is_last)}_{self.kv_args.pp_rank}"
            decode_tp_size = self.decode_kv_args_table[req.agent_name].decode_tp_size
            decode_info = self.decode_kv_args_table[req.agent_name]

            # Check if Triton transfer is enabled and supported
            prefill_tp_size = self.attn_tp_size
            use_triton = (
                self.kv_transfer_method == "triton"
                and decode_info.dst_pinned_ptr != 0
                and self.kv_args.k_buffers is not None
                and self.kv_args.v_buffers is not None
                and not self.is_mla_backend  # MLA backend not supported yet
            )

            # Triton path: only handles prefill_tp >= decode_tp cases here
            # (prefill_tp < decode_tp is handled by the batched path above)
            if use_triton and prefill_tp_size >= decode_tp_size:
                num_kv_heads = self.kv_args.kv_head_num
                local_tp_rank = self.kv_args.engine_rank % prefill_tp_size

                if prefill_tp_size > decode_tp_size:
                    # Prefill TP > Decode TP: multiple prefill ranks send to one decode rank
                    # Each prefill rank writes at a different offset in decode's buffer
                    head_start = 0
                    num_heads_to_send = num_kv_heads
                    dst_head_offset = local_tp_rank * num_kv_heads
                else:
                    # Equal TP: 1:1 mapping, each prefill rank sends to its corresponding decode rank
                    head_start = 0
                    num_heads_to_send = num_kv_heads
                    dst_head_offset = 0

                logger.debug(
                    f"[TRITON-KV] Using Triton transfer: room={bootstrap_room}, "
                    f"tokens={len(kv_indices)}, heads=[{head_start}:{head_start + num_heads_to_send}], "
                    f"dst_head_offset={dst_head_offset}, "
                    f"prefill_tp={prefill_tp_size}, decode_tp={decode_tp_size}"
                )

                # send_kvcache_triton returns (handle, pool, offset)
                kv_xfer_handle, pool, offset = self.send_kvcache_triton(
                    peer_name=req.agent_name,
                    prefill_kv_indices=kv_indices,
                    dst_pinned_ptr=decode_info.dst_pinned_ptr,
                    dst_pinned_size=decode_info.dst_pinned_size,
                    notif=notif,
                    head_start=head_start,
                    num_heads_to_send=num_heads_to_send,
                    dst_head_offset=dst_head_offset,
                )
                pool_allocations.append((pool, offset))
            elif self.is_mla_backend or (decode_tp_size == self.attn_tp_size):
                kv_xfer_handle = self.send_kvcache(
                    req.agent_name,
                    kv_indices,
                    decode_info.dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    decode_info.gpu_id,
                    notif,
                )
            else:
                # Mixed TP sizes require Triton transfer - legacy is too slow
                if self.kv_transfer_method == "triton":
                    # Triton was requested but couldn't be used - provide detailed error
                    reasons = []
                    if decode_info.dst_pinned_ptr == 0:
                        reasons.append("receiver has no pinned buffer (dst_pinned_ptr=0)")
                    if self.kv_args.k_buffers is None:
                        reasons.append("sender k_buffers is None")
                    if self.kv_args.v_buffers is None:
                        reasons.append("sender v_buffers is None")
                    raise RuntimeError(
                        f"[TRITON-KV] Mixed TP transfer (sender_tp={prefill_tp_size}, "
                        f"receiver_tp={decode_tp_size}) requires Triton transfer, but it "
                        f"could not be enabled. Reasons: {', '.join(reasons)}. "
                        f"Legacy send_kvcache_slice is not supported for mixed TP."
                    )
                # Legacy path - only allowed for same TP size (handled above)
                # This branch should not be reachable for mixed TP when triton is requested
                kv_xfer_handle = self.send_kvcache_slice(
                    req.agent_name,
                    kv_indices,
                    decode_info.dst_kv_ptrs,
                    chunked_dst_kv_indice,
                    decode_info.gpu_id,
                    notif,
                    prefill_tp_size=self.attn_tp_size,
                    decode_tp_size=decode_tp_size,
                    decode_tp_rank=decode_info.decode_tp_rank,
                    dst_kv_item_len=decode_info.dst_kv_item_len,
                )

            handles.append(kv_xfer_handle)
            # Only the last chunk we need to send the aux data.
            if is_last and self.pp_group.is_last_rank and has_aux_data:
                assert aux_index is not None
                aux_xfer_handle = self.send_aux(
                    req.agent_name,
                    aux_index,
                    decode_info.dst_aux_ptrs,
                    req.dst_aux_index,
                    str(req.room) + "_aux",
                )
                handles.append(aux_xfer_handle)
        if is_last:
            del self.transfer_infos[bootstrap_room]

        return handles, pool_allocations

    def update_transfer_status(self):
        # Process notifications from received transfers.
        notif_map = self.agent.get_new_notifs()
        for peer_name, messages in notif_map.items():
            # We could also check that self.bootstrap_info['agent_name'] matches
            # the message sender. But the bootstrap room alone should be
            # sufficient to map the status.
            for msg in messages:
                components = msg.decode("ascii").split("_", 4)
                room = int(components[0])
                if components[1] == "kv":
                    chunk_id = int(components[2])
                    is_last = bool(int(components[3]))
                    pp_rank = int(components[4]) if len(components) > 4 else 0
                    sender_key = peer_name
                    # Track received chunks per sender
                    self.transfer_statuses[room].received_kvs_per_sender[
                        sender_key
                    ].add(chunk_id)
                    if is_last:
                        # Record expected chunk count for this sender
                        self.transfer_statuses[room].expected_kvs_per_sender[
                            sender_key
                        ] = chunk_id + 1
                        # Set num_senders_expected from table (or default to 1)
                        if self.transfer_statuses[room].num_senders_expected is None:
                            self.transfer_statuses[room].num_senders_expected = (
                                self.required_prefill_response_num_table.get(room, 1)
                            )
                elif components[1] == "aux":
                    self.transfer_statuses[room].received_aux = True

    def check_transfer_done(self, room: int):
        if room not in self.transfer_statuses:
            return False
        return self.transfer_statuses[room].is_done()

    def _start_bootstrap_thread(self):
        def bootstrap_thread():
            """This thread recvs transfer info from the receiver (decode engine)"""
            while True:
                waiting_req_bytes = self.server_socket.recv_multipart()
                logger.debug(
                    f"Received multipart with total byte size {sum(len(x) for x in waiting_req_bytes)}"
                )
                assert (
                    waiting_req_bytes[0] == GUARD
                ), f"First message should be {GUARD}. Foreign traffic?"
                waiting_req_bytes = waiting_req_bytes[1:]
                room = waiting_req_bytes[0].decode("ascii")
                agent_name = waiting_req_bytes[3].decode("ascii")
                if room == "None":
                    # Register new peer and save KV base pointers.
                    self._add_remote_peer(
                        KVArgsRegisterInfo.from_zmq(waiting_req_bytes)
                    )
                    logger.debug(f"Register KVArgs from {agent_name} successfully")
                    continue
                room = int(room)
                if room not in self.transfer_infos:
                    self.transfer_infos[room] = {}
                self.transfer_infos[room][agent_name] = TransferInfo.from_zmq(
                    waiting_req_bytes
                )
                required_dst_info_num = self.transfer_infos[room][
                    agent_name
                ].required_dst_info_num
                role = (
                    "prefill"
                    if self.disaggregation_mode == DisaggregationMode.PREFILL
                    else "decode"
                )
                logger.debug(
                    "[disagg-bootstrap] %s recv transfer info room=%s agent=%s got=%s/%s",
                    role,
                    room,
                    agent_name,
                    len(self.transfer_infos[room]),
                    required_dst_info_num,
                )
                if len(self.transfer_infos[room]) == required_dst_info_num:
                    logger.debug(
                        "[disagg-bootstrap] %s room=%s bootstrapped required=%s",
                        role,
                        room,
                        required_dst_info_num,
                    )
                    self.update_status(room, KVPoll.WaitingForInput)

        threading.Thread(target=bootstrap_thread).start()


class NixlKVSender(CommonKVSender):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: int,
        dest_tp_ranks: List[int],
        pp_rank: int,
    ):
        super().__init__(mgr, bootstrap_addr, bootstrap_room, dest_tp_ranks, pp_rank)
        self.xfer_handles = []
        # Track pool allocations for release when transfer completes
        # List of (pool, offset) tuples
        self._pool_allocations: List[tuple] = []
        self.has_sent = False
        self.chunk_id = 0

    def send(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List[int]] = None,
    ):
        logger.debug(
            f"NixlKVSender.send: room={self.bootstrap_room}, "
            f"kv_indices_len={len(kv_indices)}, "
            f"curr_idx={self.curr_idx}, "
            f"num_kv_indices={self.num_kv_indices}, "
            f"chunk_id={self.chunk_id}"
        )
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last = self.curr_idx == self.num_kv_indices

        new_xfer_handles, new_pool_allocations = self.kv_mgr.add_transfer_request(
            self.bootstrap_room,
            kv_indices,
            index_slice,
            is_last,
            self.chunk_id,
            self.aux_index,
        )
        self.xfer_handles.extend(new_xfer_handles)
        self._pool_allocations.extend(new_pool_allocations)
        self.chunk_id += 1
        if is_last:
            self.has_sent = True
            del self.kv_mgr.request_status[self.bootstrap_room]

    def poll(self) -> KVPoll:
        if not self.has_sent:
            return self.kv_mgr.check_status(self.bootstrap_room)
        states = [self.kv_mgr.agent.check_xfer_state(x) for x in self.xfer_handles]
        if all([x == "DONE" for x in states]):
            # Release pool allocations now that all transfers are complete
            for pool, offset in self._pool_allocations:
                pool.release(offset)
            self._pool_allocations.clear()
            return KVPoll.Success  # type: ignore
        if any([x == "ERR" for x in states]):
            # Release pool allocations on error too
            for pool, offset in self._pool_allocations:
                pool.release(offset)
            self._pool_allocations.clear()
            raise Exception("KVSender transfer encountered an error.")
        return KVPoll.WaitingForInput  # type: ignore

    def failure_exception(self):
        raise RuntimeError("NIXL KVSender Exception")


class NixlKVReceiver(CommonKVReceiver):
    def __init__(
        self,
        mgr: NixlKVManager,
        bootstrap_addr: str,
        bootstrap_room: Optional[int] = None,
        prefill_dp_rank: Optional[int] = None,
    ):
        self.started_transfer = False
        self.conclude_state = None
        super().__init__(mgr, bootstrap_addr, bootstrap_room, prefill_dp_rank)

        # Track this room with its bootstrap address for heartbeat monitoring
        if hasattr(self.kv_mgr, "addr_to_rooms_tracker"):
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].add(
                self.bootstrap_room
            )
        self.init_time = None
        self.last_wait_log_time = 0.0
        # Store kv_indices for Triton scatter after transfer completes
        self._triton_kv_indices: Optional[npt.NDArray[np.int32]] = None
        self._triton_scatter_done = False

    def init(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
    ):
        logger.debug(
            f"NixlKVReceiver.init: room={self.bootstrap_room}, "
            f"kv_indices_len={len(kv_indices)}, "
            f"aux_index={aux_index}, "
            f"bootstrap_addr={self.bootstrap_addr}, "
            f"engine_rank={getattr(self.kv_mgr.kv_args, 'engine_rank', None)}, "
            f"prefill_dp_rank={self.prefill_dp_rank}"
        )
        if self.bootstrap_infos is None:
            logger.error(
                f"Could not fetch prefill parallel info from bootstrap_addr: {self.bootstrap_addr}",
            )
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return

        for bootstrap_info in self.bootstrap_infos:
            logger.debug(
                f"Fetched bootstrap info: {bootstrap_info} for engine rank: {self.kv_mgr.kv_args.engine_rank}"
            )
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            is_dummy = bootstrap_info["is_dummy"]
            logger.debug(
                f"Sending to sender with bootstrap room {self.bootstrap_room} {is_dummy=}"
            )
            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        str(self.bootstrap_room).encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        kv_indices.tobytes() if not is_dummy else b"",
                        str(aux_index).encode("ascii"),
                        str(self.required_dst_info_num).encode("ascii"),
                    ]
                )

        self.started_transfer = True
        self.init_time = time.time()

        # Store kv_indices for Triton scatter after transfer completes
        if self.kv_mgr.kv_transfer_method == "triton":
            self._triton_kv_indices = kv_indices.copy()
            self._triton_scatter_done = False

    def poll(self) -> KVPoll:
        if self.conclude_state is not None:
            return self.conclude_state
        status = self.kv_mgr.check_status(self.bootstrap_room)
        if status in (KVPoll.Success, KVPoll.Failed):
            self.conclude_state = status
            return status
        if not self.started_transfer:
            return KVPoll.WaitingForInput  # type: ignore

        now = time.time()
        elapsed = now - self.init_time

        if elapsed >= self.kv_mgr.waiting_timeout:
            logger.error(f"Request {self.bootstrap_room} waiting_timeout")
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"Request {self.bootstrap_room} timed out after {elapsed:.1f}s in KVPoll.WaitingForInput",
            )
            self.conclude_state = KVPoll.Failed
            return KVPoll.Failed

        self.kv_mgr.update_transfer_status()
        if self.kv_mgr.check_transfer_done(self.bootstrap_room):  # type: ignore
            self.kv_mgr.addr_to_rooms_tracker[self.bootstrap_addr].discard(
                self.bootstrap_room
            )
            # Check if the transfer failed
            if self.kv_mgr.transfer_statuses[self.bootstrap_room].is_failed():
                self.conclude_state = KVPoll.Failed
                logger.error(
                    f"NixlKVReceiver.transfer_done: room={self.bootstrap_room} FAILED "
                    f"elapsed_s={elapsed} due to node failure"
                )
            else:
                # For Triton transfer, scatter received data from pinned buffer to GPU KV cache
                if (
                    self.kv_mgr.kv_transfer_method == "triton"
                    and self._triton_kv_indices is not None
                    and not self._triton_scatter_done
                ):
                    try:
                        # Calculate head parameters for scatter
                        # The receiver gets data with head slicing already applied by sender
                        # So we scatter to head_start=0 with all received heads
                        self.kv_mgr.scatter_received_kv(
                            kv_indices=self._triton_kv_indices,
                            head_start=0,
                            num_heads_received=None,  # Use all heads in the buffer
                        )
                        self._triton_scatter_done = True
                        logger.debug(
                            f"[TRITON-KV] Scatter complete: room={self.bootstrap_room}, "
                            f"tokens={len(self._triton_kv_indices)}"
                        )
                    except Exception as e:
                        logger.error(
                            f"[TRITON-KV] Scatter failed: room={self.bootstrap_room}, error={e}"
                        )
                        self.conclude_state = KVPoll.Failed
                        del self.kv_mgr.transfer_statuses[self.bootstrap_room]
                        return KVPoll.Failed

                self.conclude_state = KVPoll.Success
                logger.debug(
                    f"NixlKVReceiver.transfer_done: room={self.bootstrap_room} SUCCESS "
                    f"elapsed_s={elapsed}"
                )
            del self.kv_mgr.transfer_statuses[self.bootstrap_room]
            return self.conclude_state  # type: ignore
        return KVPoll.WaitingForInput  # type: ignore

    def _register_kv_args(self):
        for bootstrap_info in self.bootstrap_infos:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            packed_kv_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
            )
            packed_aux_data_ptrs = b"".join(
                struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
            )

            # Get pinned buffer info for Triton KV transfer (from shared pool)
            pinned_ptr = 0
            pinned_size = 0
            if (
                self.kv_mgr.kv_transfer_method == "triton"
                and self.kv_mgr._pinned_pool is not None
            ):
                pinned_ptr, pinned_size = self.kv_mgr._pinned_pool.get_buffer_info()

            with lock:
                sock.send_multipart(
                    [
                        GUARD,
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.kv_mgr.agent.name.encode("ascii"),
                        self.kv_mgr.agent.get_agent_metadata(),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        str(self.kv_mgr.kv_args.gpu_id).encode("ascii"),
                        str(self.kv_mgr.kv_args.decode_tp_size).encode("ascii"),
                        str(self.kv_mgr.kv_args.engine_rank).encode("ascii"),
                        str(self.kv_mgr.kv_args.kv_item_lens[0]).encode("ascii"),
                        str(pinned_ptr).encode("ascii"),
                        str(pinned_size).encode("ascii"),
                    ]
                )
            logger.debug(f"Sent KV args to sender at {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')}")

    def failure_exception(self):
        raise RuntimeError("NIXL KVReceiver Exception")


class NixlKVBootstrapServer(CommonKVBootstrapServer):
    pass
