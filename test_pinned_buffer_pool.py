"""Unit tests for PinnedBufferPool."""

import threading
import time
import unittest

import torch

from python.sglang.srt.disaggregation.nixl.pinned_buffer_pool import PinnedBufferPool


class TestPinnedBufferPool(unittest.TestCase):
    def setUp(self):
        """Clear singleton instances before each test."""
        PinnedBufferPool.clear_instances()

    def tearDown(self):
        """Clear singleton instances after each test."""
        PinnedBufferPool.clear_instances()

    def test_singleton_same_gpu(self):
        """get_or_create returns same instance for same GPU."""
        pool1 = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)
        pool2 = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)
        self.assertIs(pool1, pool2)

    def test_singleton_different_gpus(self):
        """get_or_create returns different instances for different GPUs."""
        pool0 = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)
        pool1 = PinnedBufferPool.get_or_create(1, torch.float16, 1024 * 1024)
        self.assertIsNot(pool0, pool1)

    def test_basic_allocation(self):
        """Basic allocate and release works."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)  # 1MB
        offset, buf = pool.allocate(1024)  # 1KB
        self.assertEqual(offset, 0)
        self.assertIsInstance(buf, torch.Tensor)
        self.assertTrue(buf.is_pinned())
        pool.release(offset)

    def test_sequential_allocations(self):
        """Multiple allocations get different offsets."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)

        offset1, buf1 = pool.allocate(1024)
        offset2, buf2 = pool.allocate(1024)
        offset3, buf3 = pool.allocate(1024)

        # Offsets should be different (aligned to 256 bytes)
        self.assertEqual(offset1, 0)
        self.assertGreater(offset2, offset1)
        self.assertGreater(offset3, offset2)

        pool.release(offset1)
        pool.release(offset2)
        pool.release(offset3)

    def test_first_fit_allocation(self):
        """Released space is reused with first-fit strategy."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)

        # Allocate three blocks
        offset1, _ = pool.allocate(256)  # offset 0
        offset2, _ = pool.allocate(256)  # offset 256
        offset3, _ = pool.allocate(256)  # offset 512

        # Release middle block
        pool.release(offset2)

        # New allocation should reuse the gap (first-fit)
        offset4, _ = pool.allocate(256)
        self.assertEqual(offset4, offset2)

        pool.release(offset1)
        pool.release(offset3)
        pool.release(offset4)

    def test_allocation_at_start_after_release(self):
        """Releasing first block allows allocation at start."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)

        offset1, _ = pool.allocate(256)
        offset2, _ = pool.allocate(256)

        # Release first block
        pool.release(offset1)

        # New allocation should go at start
        offset3, _ = pool.allocate(256)
        self.assertEqual(offset3, 0)

        pool.release(offset2)
        pool.release(offset3)

    def test_allocation_alignment(self):
        """Allocations are aligned to 256 bytes."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)

        # Request odd size
        offset1, _ = pool.allocate(100)
        offset2, _ = pool.allocate(100)

        # Second offset should be aligned
        self.assertEqual(offset2 % 256, 0)
        self.assertEqual(offset2, 256)  # 100 -> aligned to 256

        pool.release(offset1)
        pool.release(offset2)

    def test_buffer_full_timeout(self):
        """Allocation times out when buffer is full (with explicit timeout)."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024)  # Small buffer

        # Fill the buffer
        offset, _ = pool.allocate(1024)

        # Try to allocate more - should timeout (must pass explicit timeout)
        with self.assertRaises(RuntimeError) as ctx:
            pool.allocate(256, timeout=0.1)

        self.assertIn("Timeout", str(ctx.exception))
        pool.release(offset)

    def test_buffer_full_then_released(self):
        """Allocation succeeds after space is released by another thread."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024)

        # Fill the buffer
        offset1, _ = pool.allocate(1024)

        result = {"offset": None, "error": None}

        def allocate_thread():
            try:
                offset, _ = pool.allocate(256, timeout=2.0)
                result["offset"] = offset
            except Exception as e:
                result["error"] = e

        # Start thread that will block
        t = threading.Thread(target=allocate_thread)
        t.start()

        # Give thread time to start waiting
        time.sleep(0.1)

        # Release space
        pool.release(offset1)

        # Thread should complete
        t.join(timeout=2.0)

        self.assertIsNone(result["error"])
        self.assertIsNotNone(result["offset"])

        pool.release(result["offset"])

    def test_get_stats(self):
        """get_stats returns correct allocation statistics."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)

        stats = pool.get_stats()
        self.assertEqual(stats["gpu_id"], 0)
        self.assertEqual(stats["total_bytes"], 1024 * 1024)
        self.assertEqual(stats["allocated_bytes"], 0)
        self.assertEqual(stats["num_allocations"], 0)

        offset, _ = pool.allocate(1024)

        stats = pool.get_stats()
        self.assertGreater(stats["allocated_bytes"], 0)
        self.assertEqual(stats["num_allocations"], 1)

        pool.release(offset)

        stats = pool.get_stats()
        self.assertEqual(stats["allocated_bytes"], 0)
        self.assertEqual(stats["num_allocations"], 0)

    def test_get_buffer_info(self):
        """get_buffer_info returns valid pointer and size."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)

        ptr, nbytes = pool.get_buffer_info()
        self.assertGreater(ptr, 0)
        self.assertEqual(nbytes, 1024 * 1024)

    def test_release_unknown_offset_warns(self):
        """Releasing unknown offset logs warning but doesn't crash."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024 * 1024)

        # Should not raise, just warn
        pool.release(99999)

    def test_concurrent_allocations(self):
        """Multiple threads can allocate and release concurrently."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 10 * 1024 * 1024)  # 10MB

        errors = []

        def worker(worker_id):
            try:
                for _ in range(10):
                    offset, buf = pool.allocate(1024)
                    # Do some work with buffer
                    buf.fill_(worker_id)
                    time.sleep(0.001)
                    pool.release(offset)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

        # All allocations should be released
        stats = pool.get_stats()
        self.assertEqual(stats["num_allocations"], 0)

    def test_fragmentation_handling(self):
        """Pool handles fragmented allocations correctly."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 4096)

        # Allocate 4 blocks of 256 bytes (aligned)
        offsets = []
        for _ in range(4):
            offset, _ = pool.allocate(256)
            offsets.append(offset)
        # offsets = [0, 256, 512, 768]

        # Release alternating blocks (creates fragmentation)
        pool.release(offsets[0])  # Release block at 0
        pool.release(offsets[2])  # Release block at 512

        # Allocate small block - should go to first gap (offset 0)
        new_offset, _ = pool.allocate(256)
        self.assertEqual(new_offset, 0)

        # Release block at 256 - now we have contiguous gap from 256-768
        pool.release(offsets[1])

        # Allocate 512 bytes - should fit in gap at 256-768
        large_offset, _ = pool.allocate(512)
        self.assertEqual(large_offset, 256)

        # Cleanup
        pool.release(new_offset)
        pool.release(offsets[3])
        pool.release(large_offset)


class TestPinnedBufferPoolErrorHandling(unittest.TestCase):
    """Tests for error handling and edge cases."""

    def setUp(self):
        PinnedBufferPool.clear_instances()

    def tearDown(self):
        PinnedBufferPool.clear_instances()

    def test_allocation_timeout_raises_runtime_error(self):
        """Verify timeout raises RuntimeError with helpful message."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024)

        # Fill the buffer
        offset, _ = pool.allocate(1024)

        # Attempt allocation should raise RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            pool.allocate(256, timeout=0.1)

        error_msg = str(ctx.exception)
        self.assertIn("Timeout", error_msg)
        self.assertIn("pinned-buffer-max-gb", error_msg)

        pool.release(offset)

    def test_partial_batch_allocation_failure(self):
        """Simulate what happens in conn.py when one allocation in a batch fails.

        This demonstrates the memory leak issue: if we allocate multiple regions
        and one fails, prior allocations are not automatically released.
        """
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024)

        # Simulate batch allocation like conn.py does
        pool_allocations = []

        # First allocation succeeds
        offset1, buf1 = pool.allocate(512)
        pool_allocations.append((pool, offset1))

        # Second allocation succeeds
        offset2, buf2 = pool.allocate(256)
        pool_allocations.append((pool, offset2))

        # Third allocation will fail (not enough space, only 256 bytes left)
        with self.assertRaises(RuntimeError):
            pool.allocate(512, timeout=0.1)

        # At this point, pool_allocations has 2 entries that are NOT released
        # This is the bug in conn.py - caller must handle cleanup
        stats = pool.get_stats()
        self.assertEqual(stats["num_allocations"], 2)  # Leaked!

        # Proper cleanup would be:
        for p, offset in pool_allocations:
            p.release(offset)

        stats = pool.get_stats()
        self.assertEqual(stats["num_allocations"], 0)

    def test_allocation_with_zero_timeout(self):
        """Zero timeout should fail immediately if buffer is full."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024)
        offset, _ = pool.allocate(1024)

        start = time.time()
        with self.assertRaises(RuntimeError):
            pool.allocate(256, timeout=0.0)
        elapsed = time.time() - start

        # Should fail almost immediately (< 0.1 seconds)
        self.assertLess(elapsed, 0.1)
        pool.release(offset)

    def test_infinite_wait_succeeds_when_space_released(self):
        """Default infinite wait succeeds when another thread releases space."""
        pool = PinnedBufferPool.get_or_create(0, torch.float16, 1024)

        # Fill the buffer
        offset1, _ = pool.allocate(1024)

        result = {"offset": None, "error": None}

        def allocate_thread():
            try:
                # No timeout = wait indefinitely (default behavior)
                offset, _ = pool.allocate(256)
                result["offset"] = offset
            except Exception as e:
                result["error"] = e

        # Start thread that will wait
        t = threading.Thread(target=allocate_thread)
        t.start()

        # Give thread time to start waiting
        time.sleep(0.05)

        # Release space - waiting thread should succeed
        pool.release(offset1)

        # Thread should complete quickly
        t.join(timeout=1.0)
        self.assertFalse(t.is_alive(), "Thread should have completed")

        self.assertIsNone(result["error"])
        self.assertIsNotNone(result["offset"])

        pool.release(result["offset"])


if __name__ == "__main__":
    unittest.main()
