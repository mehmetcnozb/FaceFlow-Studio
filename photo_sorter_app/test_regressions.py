"""Exercise UI callbacks and cleanup without GPU or a Tk display."""
import ast
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

TREE = ast.parse(Path(__file__).with_name('main.py').read_text(encoding='utf-8'))
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'PhotoSorterApp')

def method(name, **namespace):
    node = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<isolated method>', 'exec'), namespace)
    return namespace[name]

class RegressionTests(unittest.TestCase):
    def test_auth_error_is_available_to_delayed_callback(self):
        callbacks = []
        app = MagicMock()
        app.drive_manager.authenticate.side_effect = ValueError('login failed')
        app._post_ui.side_effect = callbacks.append
        box = MagicMock()
        method('_authenticate_drive', messagebox=box)(app)
        for callback in callbacks:
            callback()
        box.showerror.assert_called_once_with('Bağlantı Hatası', 'login failed')

    def test_failure_and_cancel_do_not_force_full_progress(self):
        for failed, cancelled in [(True, False), (False, True), (False, False)]:
            app = MagicMock()
            app.cancel_requested, app.stat_errors = cancelled, 0
            method('_finish_processing')(app, failed)
            self.assertFalse(app.is_processing)
            if failed or cancelled:
                app.progress_bar.set.assert_not_called()
            else:
                app.progress_bar.set.assert_called_once_with(1.0)

    def test_existing_temp_directory_survives_download_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            existing = root / '__temp_downloads__'
            existing.mkdir()
            sentinel = existing / 'keep.txt'
            sentinel.write_text('keep')
            app = MagicMock()
            app.output_dir, app.cancel_requested, app.stat_scanned = temp, False, 0
            app.drive_manager.list_images_in_folder.return_value = [{'name': 'photo.jpg', 'id': 'id'}]
            app.drive_manager.download_file.side_effect = OSError('network failed')
            method('_process_from_drive', tempfile=tempfile, Path=Path)(app, .45)
            app._record_error.assert_called_once()
            self.assertEqual(sentinel.read_text(), 'keep')
            self.assertEqual(list(root.iterdir()), [existing])

if __name__ == '__main__':
    unittest.main()
