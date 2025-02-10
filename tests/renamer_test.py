import unittest
from pathlib import Path

from pdf_renamer.renamer import Renamer


class Test(unittest.TestCase):
    def setUp(self):
        self._examples_path = Path(__file__).parent / "examples"
        self._model = "llama3.2"
        self._out_path = self._examples_path / "out"
        return super().setUp()

    def test_WIRED(self):
        date_format = "%Y%m%d"
        renamer = Renamer(
            model=self._model, date_format=date_format, out_path=self._out_path
        )
        in_path = (
            self._examples_path
            / "Chinese AI App DeepSeek Soars in Popularity, Startling Rivals _ WIRED.pdf"
        )
        renamed_path = renamer.rename(in_path)
        self.assertIsNotNone(renamed_path)
        self.assertEqual(
            renamed_path.name,
            "20250127_WIRED_Chinese-AI-App-DeepSeek-Soars-in-Popularity,-Startling-Rivals.pdf",
        )

    def test_New_York_Times(self):
        date_format = "%Y%m%d"
        renamer = Renamer(
            model=self._model, date_format=date_format, out_path=self._out_path
        )
        in_path = (
            self._examples_path
            / "China Is at Heart of Trump Tariffs on Steel and Aluminum - The New York Times.pdf"
        )
        renamed_path = renamer.rename(in_path)
        self.assertIsNotNone(renamed_path)
        self.assertEqual(
            renamed_path.name,
            "20250210_The-New-York-Times_China-Is-at-Heart-of-Trump-Tariffs-on-Steel-and-Aluminum.pdf",
        )

    def test_San_Francisco_Chronicle(self):
        date_format = "%Y%m%d"
        renamer = Renamer(
            model=self._model, date_format=date_format, out_path=self._out_path
        )
        in_path = (
            self._examples_path
            / "It takes time to save for a home in the Bay. But not as long as here.pdf"
        )
        renamed_path = renamer.rename(in_path)
        self.assertIsNotNone(renamed_path)
        self.assertRegex(
            renamed_path.name,
            r"20250210_no-publisher_It-takes-time-to-save-for-a-home-in-the-Bay[\w-]*.pdf",
        )

    def test_FTC(self):
        date_format = "%Y%m%d"
        renamer = Renamer(
            model=self._model, date_format=date_format, out_path=self._out_path
        )
        in_path = (
            self._examples_path
            / "p251201antitrustguidelinesbusinessactivitiesaffectingworkers2025.pdf"
        )
        renamed_path = renamer.rename(in_path)
        self.assertIsNotNone(renamed_path)
        self.assertRegex(
            renamed_path.name,
            r"20250101_no-publisher_\w*antitrustguidelinesbusinessactivitiesaffectingworkers2025.pdf",
        )

    def test_The_Washington_Post(self):
        date_format = "%Y%m%d"
        renamer = Renamer(
            model=self._model, date_format=date_format, out_path=self._out_path
        )
        in_path = (
            self._examples_path
            / "What is the CFPB, the consumer watchdog targeted by Trump_ - The Washington Post.pdf"
        )
        renamed_path = renamer.rename(in_path)
        self.assertIsNotNone(renamed_path)
        self.assertRegex(
            renamed_path.name,
            r"\w+_The-Washington-Post_What-is-the-CFPB,[\w-]+(?!The Washington Post).pdf",
        )

    def test_Mainichi(self):
        date_format = "%Y%m%d"
        renamer = Renamer(
            model=self._model, date_format=date_format, out_path=self._out_path
        )
        in_path = (
            self._examples_path
            / "夏の参院選和歌山選挙区　自民、二階氏の三男を擁立へ　残る火種とは _ 毎日新聞.pdf"
        )
        renamed_path = renamer.rename(in_path)
        self.assertIsNotNone(renamed_path)
        self.assertEqual(
            renamed_path.name,
            "20250209__毎日新聞_夏の参院選和歌山選挙区-自民、二階氏の三男を擁立へ-残る火種とは.pdf",
        )
