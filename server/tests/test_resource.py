"""Unit tests for api.resource: the compute scheduler's job/worker lifecycle,
capacity math, and /proc /sys memory-limit parsing.

api.gpu imports torch at module scope; resource.py only ever touches
gpu.device (a plain string) and gpu.total_vram()/available_vram(), so a
lightweight fake is installed into sys.modules before api.resource is
imported anywhere, avoiding a torch dependency for these tests entirely.
"""
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import MagicMock, patch
from uuid import uuid4

_fake_gpu = types.ModuleType("api.gpu")
_fake_gpu.device = "cpu"
_fake_gpu.total_vram = MagicMock(return_value=0)
_fake_gpu.available_vram = MagicMock(return_value=0)
sys.modules.setdefault("api.gpu", _fake_gpu)

from api import resource  # noqa: E402  (import must follow the gpu stub above)
from api.resource import (  # noqa: E402
    DuplicateJobId,
    InvalidJobId,
    JobCancelled,
    ManagedJob,
    ResourceProfile,
    Scheduler,
    WorkerKey,
    available_ram,
    force_stop_process,
    total_memory_limit,
)


def new_job_id() -> str:
    return str(uuid4())


class TestResourceProfileForDevice(TestCase):
    def test_cuda_device_profile_is_unchanged(self):
        profile = ResourceProfile(resident_ram=1, resident_vram=2, active_ram=3, active_vram=4, cpu_slots=5)
        assert profile.for_device("cuda:0") is profile

    def test_cpu_device_folds_vram_into_ram(self):
        profile = ResourceProfile(resident_ram=10, resident_vram=20, active_ram=1, active_vram=2, cpu_slots=3)
        folded = profile.for_device("cpu")
        assert folded.resident_ram == 30
        assert folded.resident_vram == 0
        assert folded.active_ram == 3
        assert folded.active_vram == 0
        assert folded.cpu_slots == 3


class TestWorkerKey(TestCase):
    def test_equal_keys_with_same_configuration_are_equal(self):
        a = WorkerKey(service="tts", model="foo", configuration=(("voice", "a"),))
        b = WorkerKey(service="tts", model="foo", configuration=(("voice", "a"),))
        assert a == b
        assert hash(a) == hash(b)

    def test_different_configuration_is_a_different_key(self):
        a = WorkerKey(service="tts", model="foo", configuration=(("voice", "a"),))
        b = WorkerKey(service="tts", model="foo", configuration=(("voice", "b"),))
        assert a != b


class TestJobCancelled(TestCase):
    def test_cancelled_code_has_generic_message(self):
        error = JobCancelled("cancelled")
        assert error.code == "cancelled"
        assert "cancelled" in str(error)

    def test_heartbeat_expired_code_has_specific_message(self):
        error = JobCancelled("heartbeat_expired")
        assert error.code == "heartbeat_expired"
        assert "heartbeat" in str(error)


class TestSchedulerRegister(TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.scheduler._stopping = True
        with self.scheduler.changed:
            self.scheduler.changed.notify_all()

    def test_register_with_missing_id_raises(self):
        with self.assertRaises(InvalidJobId):
            self.scheduler.register(None)

    def test_register_with_malformed_id_raises(self):
        with self.assertRaises(InvalidJobId):
            self.scheduler.register("not-a-uuid")

    def test_register_returns_a_managed_job(self):
        job_id = new_job_id()
        job = self.scheduler.register(job_id)
        assert isinstance(job, ManagedJob)
        assert job.id == job_id
        assert job_id in self.scheduler.jobs

    def test_register_duplicate_id_raises(self):
        job_id = new_job_id()
        self.scheduler.register(job_id)
        with self.assertRaises(DuplicateJobId):
            self.scheduler.register(job_id)

    def test_register_reuses_normalized_uuid_string(self):
        # UUID(raw_id) round-trips through str(), so mixed-case input
        # normalizes to the same job id as its lowercase form.
        raw = str(uuid4())
        job = self.scheduler.register(raw.upper())
        assert job.id == raw


class TestSchedulerHeartbeatAndCancel(TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.scheduler._stopping = True
        with self.scheduler.changed:
            self.scheduler.changed.notify_all()

    def test_heartbeat_on_unknown_job_returns_false(self):
        assert self.scheduler.heartbeat(new_job_id()) is False

    def test_heartbeat_on_malformed_id_returns_false(self):
        assert self.scheduler.heartbeat("not-a-uuid") is False

    def test_heartbeat_extends_deadline(self):
        job_id = new_job_id()
        job = self.scheduler.register(job_id)
        original_deadline = job.heartbeat_deadline
        job.heartbeat_deadline = 0  # force it to look overdue
        assert self.scheduler.heartbeat(job_id) is True
        assert job.heartbeat_deadline > original_deadline

    def test_cancel_unknown_job_returns_false(self):
        assert self.scheduler.cancel(new_job_id()) is False

    def test_cancel_malformed_id_returns_false(self):
        assert self.scheduler.cancel("not-a-uuid") is False

    def test_cancel_sets_cancellation_event(self):
        job_id = new_job_id()
        job = self.scheduler.register(job_id)
        assert self.scheduler.cancel(job_id) is True
        assert job.cancellation.is_set()
        assert job.cancel_code == "cancelled"

    def test_cancel_is_idempotent_on_the_cancel_code(self):
        job_id = new_job_id()
        job = self.scheduler.register(job_id)
        self.scheduler.cancel(job_id, "cancelled")
        # A second cancel call must not overwrite an already-set code.
        self.scheduler.cancel(job_id, "cancelled")
        assert job.cancel_code == "cancelled"

    def test_register_after_cancel_raises_within_grace_window(self):
        job_id = new_job_id()
        job = self.scheduler.register(job_id)
        self.scheduler.cancel(job_id)
        self.scheduler.unregister(job)
        with self.assertRaises(JobCancelled):
            self.scheduler.register(job_id)

    def test_check_cancelled_raises_with_the_right_code(self):
        job = ManagedJob(new_job_id())
        job.cancel_code = "heartbeat_expired"
        job.cancellation.set()
        with self.assertRaises(JobCancelled) as ctx:
            job.check_cancelled()
        assert ctx.exception.code == "heartbeat_expired"

    def test_check_cancelled_is_a_noop_when_not_cancelled(self):
        job = ManagedJob(new_job_id())
        job.check_cancelled()  # must not raise


class TestSchedulerUnregister(TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.scheduler._stopping = True
        with self.scheduler.changed:
            self.scheduler.changed.notify_all()

    def test_unregister_removes_the_job(self):
        job_id = new_job_id()
        job = self.scheduler.register(job_id)
        self.scheduler.unregister(job)
        assert job_id not in self.scheduler.jobs

    def test_unregister_releases_temporary_storage_reservation(self):
        job = self.scheduler.register(new_job_id())
        job.temporary_bytes = 123
        self.scheduler._temporary_bytes = 123
        self.scheduler._temporary_capacity = 1000
        self.scheduler.unregister(job)
        assert self.scheduler._temporary_bytes == 0
        assert self.scheduler._temporary_capacity is None

    def test_unregister_on_a_never_registered_job_is_safe(self):
        # job._record is None and job.id was never added to self.jobs;
        # unregister must not raise.
        job = ManagedJob(new_job_id())
        self.scheduler.unregister(job)


class TestSchedulerReserveTemporary(TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.scheduler._stopping = True
        with self.scheduler.changed:
            self.scheduler.changed.notify_all()

    def test_non_positive_size_raises(self):
        job = self.scheduler.register(new_job_id())
        with self.assertRaises(ValueError):
            self.scheduler.reserve_temporary(job, 0)
        with self.assertRaises(ValueError):
            self.scheduler.reserve_temporary(job, -1)

    def test_reserving_twice_for_the_same_job_raises(self):
        job = self.scheduler.register(new_job_id())
        with patch("shutil.disk_usage") as mock_disk_usage:
            mock_disk_usage.return_value = MagicMock(free=10**12)
            self.scheduler.reserve_temporary(job, 1024)
            with self.assertRaises(RuntimeError):
                self.scheduler.reserve_temporary(job, 1024)

    def test_reservation_exceeding_free_disk_raises(self):
        job = self.scheduler.register(new_job_id())
        with patch("shutil.disk_usage") as mock_disk_usage:
            mock_disk_usage.return_value = MagicMock(free=100)
            with self.assertRaises(resource.TemporaryStorageUnavailable):
                self.scheduler.reserve_temporary(job, 1000)

    def test_successful_reservation_tracks_bytes(self):
        job = self.scheduler.register(new_job_id())
        with patch("shutil.disk_usage") as mock_disk_usage:
            mock_disk_usage.return_value = MagicMock(free=10**12)
            self.scheduler.reserve_temporary(job, 4096)
        assert job.temporary_bytes == 4096
        assert self.scheduler._temporary_bytes == 4096

    def test_check_cancelled_before_reserving_raises(self):
        job = self.scheduler.register(new_job_id())
        self.scheduler.cancel(job.id)
        with self.assertRaises(JobCancelled):
            self.scheduler.reserve_temporary(job, 1024)


class TestSchedulerAcquireGuards(TestCase):
    """acquire()'s synchronous guard checks, without exercising the
    worker-assignment/factory machinery (covered indirectly by the
    integration-shaped tests would need a real ComputeWorker; the guard
    clauses below are pure and don't)."""

    def setUp(self):
        self.scheduler = Scheduler()
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.scheduler._stopping = True
        with self.scheduler.changed:
            self.scheduler.changed.notify_all()

    def test_acquire_on_a_cancelled_job_raises(self):
        job = self.scheduler.register(new_job_id())
        self.scheduler.cancel(job.id)
        with self.assertRaises(JobCancelled):
            self.scheduler.acquire(
                job,
                WorkerKey(service="tts", model="m"),
                ResourceProfile(),
                factory=MagicMock(),
                queued=MagicMock(),
            )

    def test_acquire_on_an_already_queued_job_raises(self):
        job = self.scheduler.register(new_job_id())
        with patch.object(Scheduler, "_possible_locked", return_value=True), \
             patch.object(Scheduler, "_schedule_locked"):
            job._record = MagicMock()
            with self.assertRaises(RuntimeError):
                self.scheduler.acquire(
                    job,
                    WorkerKey(service="tts", model="m"),
                    ResourceProfile(),
                    factory=MagicMock(),
                    queued=MagicMock(),
                )

    def test_acquire_exceeding_capacity_raises(self):
        job = self.scheduler.register(new_job_id())
        with patch.object(Scheduler, "_possible_locked", return_value=False):
            with self.assertRaises(RuntimeError):
                self.scheduler.acquire(
                    job,
                    WorkerKey(service="tts", model="m"),
                    ResourceProfile(),
                    factory=MagicMock(),
                    queued=MagicMock(),
                )


class TestSchedulerCapacityMath(TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.scheduler._stopping = True
        with self.scheduler.changed:
            self.scheduler.changed.notify_all()

    def test_possible_locked_rejects_more_cpu_slots_than_available(self):
        with patch("os.process_cpu_count", return_value=2):
            assert self.scheduler._possible_locked(ResourceProfile(cpu_slots=3)) is False

    def test_possible_locked_rejects_ram_over_the_memory_limit(self):
        with patch("os.process_cpu_count", return_value=4), \
             patch.object(resource, "total_memory_limit", return_value=100):
            profile = ResourceProfile(resident_ram=60, active_ram=60, cpu_slots=1)
            assert self.scheduler._possible_locked(profile) is False

    def test_possible_locked_allows_fitting_profile_on_cpu(self):
        with patch("os.process_cpu_count", return_value=4), \
             patch.object(resource, "total_memory_limit", return_value=1000):
            profile = ResourceProfile(resident_ram=10, active_ram=10, cpu_slots=1)
            assert self.scheduler._possible_locked(profile) is True

    def test_possible_locked_checks_vram_on_cuda_device(self):
        _fake_gpu.device = "cuda:0"
        _fake_gpu.total_vram = MagicMock(return_value=100)
        try:
            with patch("os.process_cpu_count", return_value=4), \
                 patch.object(resource, "total_memory_limit", return_value=None):
                fits = ResourceProfile(resident_vram=50, active_vram=50, cpu_slots=1)
                exceeds = ResourceProfile(resident_vram=60, active_vram=60, cpu_slots=1)
                assert self.scheduler._possible_locked(fits) is True
                assert self.scheduler._possible_locked(exceeds) is False
        finally:
            _fake_gpu.device = "cpu"

    def test_fits_delta_locked_rejects_over_cpu_capacity(self):
        self.scheduler._cpu_slots = 2
        with patch("os.process_cpu_count", return_value=4):
            assert self.scheduler._fits_delta_locked(0, 0, 3) is False

    def test_fits_delta_locked_rejects_over_memory_limit(self):
        self.scheduler._reserved_ram = 50
        with patch("os.process_cpu_count", return_value=4), \
             patch.object(resource, "total_memory_limit", return_value=100), \
             patch.object(resource, "available_ram", return_value=10**12):
            assert self.scheduler._fits_delta_locked(60, 0, 0) is False

    def test_fits_delta_locked_rejects_when_live_ram_is_scarce(self):
        with patch("os.process_cpu_count", return_value=4), \
             patch.object(resource, "total_memory_limit", return_value=None), \
             patch.object(resource, "available_ram", return_value=50):
            self.scheduler._pending_ram = 40
            assert self.scheduler._fits_delta_locked(20, 0, 0) is False

    def test_fits_delta_locked_allows_when_everything_fits(self):
        with patch("os.process_cpu_count", return_value=4), \
             patch.object(resource, "total_memory_limit", return_value=None), \
             patch.object(resource, "available_ram", return_value=10**12):
            assert self.scheduler._fits_delta_locked(10, 0, 1) is True

    def test_reserve_locked_with_include_resident_adds_all_fields(self):
        self.scheduler._reserve_locked(
            ResourceProfile(resident_ram=1, resident_vram=2, active_ram=3, active_vram=4, cpu_slots=5),
            include_resident=True,
        )
        assert self.scheduler._reserved_ram == 4
        assert self.scheduler._reserved_vram == 6
        assert self.scheduler._cpu_slots == 5

    def test_reserve_locked_without_include_resident_skips_resident_fields(self):
        self.scheduler._reserve_locked(
            ResourceProfile(resident_ram=1, resident_vram=2, active_ram=3, active_vram=4, cpu_slots=5),
            include_resident=False,
        )
        assert self.scheduler._reserved_ram == 3
        assert self.scheduler._reserved_vram == 4
        assert self.scheduler._cpu_slots == 5


class TestSchedulerSelectDevice(TestCase):
    def setUp(self):
        self.scheduler = Scheduler()
        self.addCleanup(self._shutdown)

    def _shutdown(self):
        self.scheduler._stopping = True
        with self.scheduler.changed:
            self.scheduler.changed.notify_all()

    def test_select_device_with_active_jobs_raises(self):
        self.scheduler.register(new_job_id())
        with self.assertRaises(BlockingIOError):
            self.scheduler.select_device("cpu")

    def test_select_device_to_the_same_device_is_a_noop(self):
        _fake_gpu.device = "cpu"
        self.scheduler.select_device("cpu")
        assert _fake_gpu.device == "cpu"

    def test_select_device_switches_when_idle(self):
        _fake_gpu.device = "cpu"
        try:
            self.scheduler.select_device("cuda:0")
            assert _fake_gpu.device == "cuda:0"
        finally:
            _fake_gpu.device = "cpu"


class TestSchedulerSummaryAndShutdown(TestCase):
    def setUp(self):
        self.scheduler = Scheduler()

    def test_summary_on_a_fresh_scheduler_is_empty(self):
        summary = self.scheduler.summary()
        assert summary["queued_jobs"] == 0
        assert summary["active_jobs"] == 0
        assert summary["workers"] == []

    def test_summary_counts_active_jobs_by_state(self):
        job = self.scheduler.register(new_job_id())
        job.state = "running"
        summary = self.scheduler.summary()
        assert summary["active_jobs"] == 1

    def test_shutdown_cancels_all_registered_jobs(self):
        job = self.scheduler.register(new_job_id())
        self.scheduler.shutdown()
        assert job.cancellation.is_set()
        assert job.cancel_code == "cancelled"

    def test_shutdown_is_safe_with_no_jobs(self):
        Scheduler().shutdown()


class TestForceStopProcess(TestCase):
    def test_none_process_and_no_connection_is_a_noop(self):
        force_stop_process(None)

    def test_closes_connection_even_with_no_process(self):
        connection = MagicMock()
        force_stop_process(None, connection)
        connection.close.assert_called_once()

    def test_connection_close_failure_is_raised_when_process_is_none(self):
        connection = MagicMock()
        connection.close.side_effect = OSError("boom")
        with self.assertRaises(OSError):
            force_stop_process(None, connection)

    def test_already_stopped_process_returns_cleanly(self):
        process = MagicMock()
        process.is_alive.return_value = False
        force_stop_process(process)
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_process_that_wont_die_raises(self):
        process = MagicMock()
        process.is_alive.return_value = True
        with self.assertRaises(RuntimeError):
            force_stop_process(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()


class TestMemoryLimitParsing(TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _patch_paths(self, *, meminfo=None, cgroup_max=None, cgroup_stat=None, cgroup_current=None, zfs=None):
        # Every attr is patched to a path under self.root on every call,
        # whether or not this test writes content for it: resource.py's
        # inner try/except falls through to the *next* real path (e.g.
        # CGROUP_MEMORY_CURRENT) on a read failure, and this sandbox's
        # real /sys/fs/cgroup/memory.stat is readable -- leaving any of
        # these attrs unpatched pulls in the host's live memory stats
        # instead of this test's fixture.
        patches = []
        for attr, content in (
            ("MEMINFO", meminfo),
            ("CGROUP_MEMORY_MAX", cgroup_max),
            ("CGROUP_MEMORY_STAT", cgroup_stat),
            ("CGROUP_MEMORY_CURRENT", cgroup_current),
            ("ZFS_ARCSTATS", zfs),
        ):
            path = self.root / attr
            if content is not None:
                path.write_text(content)
            patches.append(patch.object(resource, attr, path))
        return patches

    def _apply(self, patches):
        for one in patches:
            self.addCleanup(one.stop)
            one.start()

    def test_total_memory_limit_uses_cgroup_max_when_lower(self):
        self._apply(self._patch_paths(
            meminfo="MemTotal:       16777216 kB\n",
            cgroup_max="8589934592\n",  # 8 GiB, lower than the 16 GiB system total
        ))
        assert total_memory_limit() == 8589934592

    def test_total_memory_limit_falls_back_to_system_total_when_cgroup_is_max(self):
        self._apply(self._patch_paths(
            meminfo="MemTotal:       16777216 kB\n",
            cgroup_max="max\n",
        ))
        assert total_memory_limit() == 16777216 * 1024

    def test_total_memory_limit_returns_none_when_nothing_is_readable(self):
        self._apply(self._patch_paths())
        assert total_memory_limit() is None

    def test_total_memory_limit_ignores_malformed_cgroup_value(self):
        self._apply(self._patch_paths(
            meminfo="MemTotal:       1024 kB\n",
            cgroup_max="not-a-number\n",
        ))
        assert total_memory_limit() == 1024 * 1024

    def test_available_ram_uses_cgroup_stat_usage_when_present(self):
        self._apply(self._patch_paths(
            meminfo="MemTotal:       16777216 kB\nMemAvailable:   8388608 kB\n",
            cgroup_max="4294967296\n",  # 4 GiB
            cgroup_stat="anon 1073741824\nkernel 0\nshmem 0\n",  # 1 GiB used
        ))
        # cgroup_available = 4 GiB - 1 GiB = 3 GiB, below the 8 GiB
        # system-available figure, so the cgroup number wins.
        assert available_ram() == 4294967296 - 1073741824

    def test_available_ram_falls_back_to_cgroup_current_when_stat_missing(self):
        self._apply(self._patch_paths(
            meminfo="MemTotal:       16777216 kB\nMemAvailable:   8388608 kB\n",
            cgroup_max="4294967296\n",
            cgroup_current="2147483648\n",  # 2 GiB used
        ))
        assert available_ram() == 4294967296 - 2147483648

    def test_available_ram_returns_system_value_when_cgroup_is_max(self):
        self._apply(self._patch_paths(
            meminfo="MemTotal:       16777216 kB\nMemAvailable:   8388608 kB\n",
            cgroup_max="max\n",
        ))
        assert available_ram() == 8388608 * 1024

    def test_available_ram_returns_none_when_nothing_is_readable(self):
        self._apply(self._patch_paths())
        assert available_ram() is None

    def test_available_ram_folds_in_reclaimable_zfs_arc(self):
        self._apply(self._patch_paths(
            meminfo="MemTotal:       16777216 kB\nMemAvailable:   1048576 kB\n",
            cgroup_max="max\n",
            zfs="name type data\nsize 4 2097152\nc_min 4 524288\n",
        ))
        # Reclaimable ARC = size(2097152) - c_min(524288) = 1572864 bytes,
        # added to the 1 GiB MemAvailable, capped at MemTotal.
        expected = min(16777216 * 1024, 1048576 * 1024 + 1572864)
        assert available_ram() == expected
