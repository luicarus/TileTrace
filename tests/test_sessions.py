import multiprocessing
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from triton_transform.sessions import SessionStore


def publish_context(directory, version, started, read, release, done, results, clear=False):
    class PausingStore(SessionStore):
        def _read(self, session_id):
            context = super()._read(session_id)
            if read is not None:
                read.set()
                if not release.wait(10):
                    raise TimeoutError('Test did not release the paused reader')
            return context

    try:
        store = PausingStore(directory)
        started.set()
        if clear:
            store.clear('window-1')
            results.put((version, None))
        else:
            results.put((version, store.put('window-1', {'document_id': 'a', 'version': version})))
    except Exception as exc:
        results.put((version, repr(exc)))
    finally:
        done.set()


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = SessionStore(Path(self.directory.name))

    def test_round_trip_preserves_document_and_selection(self):
        context = {"document_id": "file:///kernel.py", "version": 3, "selected_node_id": "n1"}
        self.assertTrue(self.store.put("window-1", context))
        self.assertEqual(self.store.get("window-1"), context)

    def test_same_document_rejects_older_version(self):
        self.store.put("window-1", {"document_id": "a", "version": 3})
        self.assertFalse(self.store.put("window-1", {"document_id": "a", "version": 2}))
        self.assertEqual(self.store.get("window-1")["version"], 3)

    def test_new_document_may_start_at_lower_version(self):
        self.store.put("window-1", {"document_id": "a", "version": 9})
        self.store.put("window-1", {"document_id": "b", "version": 1})
        self.assertEqual(self.store.get("window-1")["document_id"], "b")

    def test_sessions_do_not_share_current_file(self):
        self.store.put("window-1", {"document_id": "a", "version": 1})
        self.store.put("window-2", {"document_id": "b", "version": 1})
        self.assertEqual(self.store.get("window-1")["document_id"], "a")
        self.assertEqual(self.store.get("window-2")["document_id"], "b")

    def test_invalid_session_ids_cannot_escape_directory(self):
        for session in ("../bad", "a/b", "a\\b", "", "x" * 81):
            with self.subTest(session=session), self.assertRaises(ValueError):
                self.store.put(session, {"document_id": "a", "version": 1})

    def test_missing_session_returns_no_context(self):
        self.assertIsNone(self.store.get("absent"))

    def test_clear_removes_only_named_session(self):
        self.store.put("window-1", {"document_id": "a", "version": 1})
        self.store.put("window-2", {"document_id": "b", "version": 1})
        self.store.clear("window-1")
        self.assertIsNone(self.store.get("window-1"))
        self.assertIsNotNone(self.store.get("window-2"))

    def test_malformed_existing_context_is_reported(self):
        self.store.root.mkdir(parents=True, exist_ok=True)
        (self.store.root / "broken.json").write_text("not json", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.get("broken")

    def test_version_must_be_an_integer(self):
        with self.assertRaises(ValueError):
            self.store.put("window-1", {"document_id": "a", "version": "3"})

    def interleaved_publication(self, processes=False, clear=False, newer_first=False):
        self.store.clear('window-1')
        self.store.put('window-1', {'document_id': 'a', 'version': 1})
        factory = multiprocessing.get_context('spawn') if processes else threading
        results = factory.Queue() if processes else queue.Queue()
        started_old, started_new, read_old, release = [factory.Event() for _ in range(4)]
        done_old, done_new = factory.Event(), factory.Event()
        runner = factory.Process if processes else factory.Thread
        first_version, second_version = (3, 2) if newer_first else (2, 3)
        old = runner(target=publish_context, args=(self.directory.name, first_version, started_old, read_old,
                                                   release, done_old, results))
        new = runner(target=publish_context, args=(self.directory.name, second_version, started_new, None,
                                                   release, done_new, results, clear))
        old.start()
        new_started = False
        try:
            self.assertTrue(read_old.wait(5), 'First publisher did not reach its previous-version read')
            new.start()
            new_started = True
            self.assertTrue(started_new.wait(5), 'Competing publisher did not start')
            completed_during_old_read = done_new.wait(0.5)
        finally:
            release.set()
            old.join(10)
            if new_started:
                new.join(10)
            for task in (old, new) if new_started else (old,):
                if processes and task.is_alive():
                    task.terminate()
                    task.join(5)
        self.assertFalse(old.is_alive())
        self.assertFalse(new.is_alive())
        outcomes = dict(results.get(timeout=5) for _ in range(2))
        if processes:
            results.close()
            results.join_thread()
            self.assertEqual(old.exitcode, 0)
            self.assertEqual(new.exitcode, 0)
        self.assertEqual(outcomes, {first_version: True,
                                   second_version: None if clear else not newer_first})
        if clear:
            self.assertIsNone(self.store.get('window-1'))
        else:
            self.assertEqual(self.store.get('window-1')['version'], 3)
        self.assertFalse(completed_during_old_read, 'Competing mutation bypassed the publication lock')

    def test_independent_stores_serialize_compare_and_replace(self):
        for newer_first in (False, True):
            with self.subTest(newer_first=newer_first):
                self.interleaved_publication(newer_first=newer_first)

    def test_independent_processes_serialize_compare_and_replace(self):
        for newer_first in (False, True):
            with self.subTest(newer_first=newer_first):
                self.interleaved_publication(processes=True, newer_first=newer_first)

    def test_clear_serializes_with_in_progress_publication(self):
        self.interleaved_publication(clear=True)

    def test_contended_session_lock_has_bounded_wait(self):
        self.store.put('window-1', {'document_id': 'a', 'version': 1})
        started, read, release, done = [threading.Event() for _ in range(4)]
        results = queue.Queue()
        publisher = threading.Thread(target=publish_context, args=(self.directory.name, 2, started,
                                                                  read, release, done, results))
        publisher.start()
        try:
            self.assertTrue(read.wait(5))
            with patch('triton_transform.sessions._LOCK_TIMEOUT_SECONDS', 0.1, create=True):
                with self.assertRaisesRegex(TimeoutError, 'session.*window-1'):
                    SessionStore(self.directory.name).put('window-1', {'document_id': 'a', 'version': 3})
        finally:
            release.set()
            publisher.join(10)
        self.assertFalse(publisher.is_alive())
        self.assertEqual(results.get(timeout=5), (2, True))

    def test_nonfinite_existing_context_is_rejected(self):
        for value in ('NaN', '1e999'):
            with self.subTest(value=value):
                (self.store.root / 'broken.json').write_text(
                    '{"document_id":"a","version":1,"value":' + value + '}', encoding='utf-8')
                with self.assertRaises(ValueError):
                    self.store.get('broken')

    def test_snapshot_reader_serializes_with_publication(self):
        self.store.put('window-1', {'document_id': 'a', 'version': 1})
        target = self.store.root / 'window-1.json'
        opened, release, started, done = [threading.Event() for _ in range(4)]
        results = queue.Queue()
        original_read_text = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == target and threading.current_thread() is reader:
                # Hold a real ordinary Python reader open. On Windows it does
                # not share delete access, so unlocked os.replace would fail.
                with path.open('r', encoding=kwargs.get('encoding')) as stream:
                    opened.set()
                    if not release.wait(10):
                        raise TimeoutError('Test did not release the snapshot reader')
                    return stream.read()
            return original_read_text(path, *args, **kwargs)

        def read_context():
            try:
                results.put(('read', SessionStore(self.directory.name).get('window-1')))
            except Exception as exc:
                results.put(('read', repr(exc)))

        def write_context():
            started.set()
            try:
                results.put(('put', SessionStore(self.directory.name).put(
                    'window-1', {'document_id': 'a', 'version': 2})))
            except Exception as exc:
                results.put(('put', repr(exc)))
            finally:
                done.set()

        reader = threading.Thread(target=read_context)
        writer = threading.Thread(target=write_context)
        writer_started = False
        with patch.object(Path, 'read_text', read_text):
            reader.start()
            try:
                self.assertTrue(opened.wait(5), 'Reader did not open the snapshot')
                writer.start()
                writer_started = True
                self.assertTrue(started.wait(5), 'Publisher did not start')
                completed_during_read = done.wait(0.5)
            finally:
                release.set()
                reader.join(10)
                if writer_started:
                    writer.join(10)
        self.assertFalse(reader.is_alive())
        self.assertFalse(writer.is_alive())
        outcomes = dict(results.get(timeout=5) for _ in range(2))
        self.assertEqual(outcomes['put'], True)
        self.assertEqual(outcomes['read'], {'document_id': 'a', 'version': 1})
        self.assertEqual(self.store.get('window-1')['version'], 2)
        self.assertFalse(completed_during_read, 'Publisher did not wait for the open snapshot reader')

    def test_absent_session_read_does_not_create_directory(self):
        store = SessionStore(self.store.root / 'absent-directory')
        self.assertIsNone(store.get('window-1'))
        self.assertFalse(store.root.exists())


if __name__ == "__main__":
    unittest.main()
