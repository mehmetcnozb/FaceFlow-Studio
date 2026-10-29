import tempfile
import unittest
from pathlib import Path
from file_safety import copy_unique


class FileSafetyTests(unittest.TestCase):
    def test_existing_photo_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.jpg'
            source.write_bytes(b'new image')
            existing = root / 'photo.jpg'
            existing.write_bytes(b'original image')
            saved = copy_unique(source, root, 'photo.jpg')
            self.assertEqual(existing.read_bytes(), b'original image')
            self.assertEqual(saved.name, 'photo (1).jpg')
            self.assertEqual(saved.read_bytes(), b'new image')

    def test_drive_name_cannot_escape_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.jpg'
            source.write_bytes(b'image')
            saved = copy_unique(source, root, '../outside.jpg')
            self.assertEqual(saved.parent.resolve(), root.resolve())

    def test_failed_copy_leaves_no_partial_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(FileNotFoundError):
                copy_unique(root / 'missing', root, 'photo.jpg')
            self.assertFalse((root / 'photo.jpg').exists())



if __name__ == '__main__':
    unittest.main()
