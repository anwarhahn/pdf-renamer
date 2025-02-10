import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

parser = argparse.ArgumentParser(
    prog="PDF Renamer",
    description="Use LLMs to rename PDFs of downloaded news articles.",
)
parser.add_argument(
    "input_dir",
    help="The directory where the to-be-renamed PDFs are saved.",
)
parser.add_argument(
    "output_dir",
    help="The directory where the newly renamed PDFs are saved.",
)
parser.add_argument("model", default="llama3.2", help="The LLM to use.")
parser.add_argument(
    "date_format",
    type=str,
    default='"%Y%m%d"',
    help="The format of the date that is prefixed to the filename. "
    "See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes",
)
parser.add_argument(
    "--logfile", help="The absolute path where you want a logfile stored."
)


class Renamer:
    def __init__(self, model: str, date_format: str, out_path: Path):
        self._model = ChatOllama(model=model)
        self._publish_date_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a robot that only returns a single JSON object.
                    
                    Extract the publish date of the article or the first date you find.
                    Format the publish date of the article in the following format: {date_format}.

                    Here are some examples:
                    Text: JAN 27, 2025 12:05 PM
                    Response: {{
                        "end_date": "2025-01-27",
                        "reasoning": "There was a date and time at the beginning of the document."
                    }}

                    Text: Revised: September 1999
                    Response: {{
                        "end_date": "1999-09-01",
                        "reasoning": "Document was revised in 'September 1999'."
                    }}

                    Text: By Keith Bradsher
                    Keith Bradsher, who started covering international trade in steel in 1991, 
                    reported from Hong Kong.
                    May. 10, 2022, 4:56 a.m. ET
                    Response: {{
                        "end_date": "2022-05-10",
                        "reasoning": "Date found close to the author's name."
                    }}

                    Text: 毎⽇新聞 2024/12/19 21:56
                    Response: {{
                        "end_date": "2024-12-19",
                        "reasoning": "Date found close to the author's name."
                    }}

                    Text: By Christian Leonard, Data Reporter
                    July 11, 2021
                    Response: {{
                        "end_date": "2021-07-11",
                        "reasoning": "Date found close to the author's name."
                    }}

                    Text: Today at 9:11 a.m. EST
                    Response: {{
                        "end_date": "",
                        "reasoning": "'Today' is a relative date."
                    }}
                    
                    In the JSON object value you return, the "end_date" field should contain either
                    the formatted end date or the empty string. In the JSON object value you return,
                    the "reasoning" field should contain a string describing why you returned this 
                    the value for "end_date".

                    """,
                ),
                ("user", "{text}"),
            ]
        )

        self._publisher_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a robot that only returns a single JSON object.
                    
                    Extract the title of the article and the name of the article's publisher by 
                    looking at the filename. The publisher name is often suffixed to the filename 
                    with hyphens or underscores.

                    Here are some examples:

                    Filename: China Is at Heart of Trump Tariffs on Steel and Aluminum - The New York Times.pdf
                    Response: {{
                        "publisher": "The New York Times",
                        "title": "China Is at Heart of Trump Tariffs on Steel and Aluminum",
                        "reasoning": "The publisher's name is separated from the title by ' - '."
                    }}
                    
                    Filename: What is the CFPB, the consumer watchdog targeted by Trump_ - The Washington Post.pdf
                    Response: {{
                        "publisher": "The Washington Post",
                        "title": "What is the CFPB, the consumer watchdog targeted by Trump",
                        "reasoning": "The publisher's name is separated from the title by '_ - '."
                    }}
                    
                    In the JSON object value you return:
                    - The "publisher" field should contain either the publisher's name or the empty 
                      string. 
                    - The "title" field should contain the substring that excludes the publisher name.
                    - The "reasoning" field should contain a string describing why you returned 
                      these values.
                    """,
                ),
                (
                    "user",
                    """
                    Filename: {filename}
                    """,
                ),
            ]
        )
        self._date_format = date_format
        self._out_path = out_path

    def rename(self, in_path: Path) -> str | None:
        logger.info('Starting rename of "%s".', str(in_path))

        # Load the first page of the PDF.
        loader = PyPDFLoader(str(in_path.resolve()))
        first_page = next(loader.lazy_load()).page_content

        # Extract the publish date using a LLM.
        publish_date = self._extract_publish_date(first_page)
        if publish_date is None:
            publish_date = "no-date"

        # Extract the publisher using a LLM.
        title, publisher = self._extract_publisher(in_path.name)
        if title is None or publisher is None:
            # Extract the publisher and title using heuristics.
            h_title, h_publisher = self._split_publisher_and_title(in_path.stem)
            title = title or h_title
            publisher = publisher or h_publisher

        # Kebab case the title.
        title = "-".join([w for w in title.split() if w])

        # Kebab case the publisher.
        publisher = "-".join([w for w in publisher.split() if w])

        # Combine the elements together to create a new filename.
        renamed = self._make_unique_filename(in_path, publish_date, publisher, title)
        logger.info('Finishing rename of "%s" -> "%s".', in_path.name, renamed)
        return renamed

    def _invoke_model(self, prompt: PromptValue) -> Any | None:
        """Use an LLM to extract the info needed."""
        message = self._model.invoke(prompt)
        logger.info("Invoked model: %s", message)
        try:
            answer = json.loads(message.content)
        except json.JSONDecodeError as e:
            logger.exception(e)
        else:
            return answer

    def _extract_publish_date(self, text: str) -> str | None:
        """Use an LLM to extract the publish date."""
        answer = self._invoke_model(
            self._publish_date_prompt_template.invoke(
                {
                    "date_format": "YYYY-MM-DD",
                    "text": text,
                }
            )
        )

        if "end_date" in answer and answer["end_date"]:
            try:
                publish_date = date.fromisoformat(answer["end_date"])
            except ValueError as e:
                logger.exception(e)
                return
            publish_date = publish_date.strftime(self._date_format)
            logger.info("Successfully extracted publish date: (%s).", publish_date)
            return publish_date
        else:
            logger.error("Date extraction failed.")

    def _extract_publisher(self, filename: str) -> tuple[str | None, str | None]:
        answer = self._invoke_model(
            self._publisher_prompt_template.invoke(
                {
                    "filename": filename,
                }
            )
        )

        title = publisher = None
        if "publisher" in answer and answer["publisher"]:
            publisher = answer["publisher"]
            logger.info("Successfully extracted publisher: (%s).", publisher)
        else:
            logger.error("Publisher extraction failed.")

        if "title" in answer and answer["title"]:
            title = answer["title"]
            logger.info("Successfully extracted title: (%s).", title)
        else:
            logger.error("Title extraction failed.")

        return title, publisher

    def _split_publisher_and_title(self, filename: str) -> tuple[str, str]:
        """Return the publisher and the title using heuristics."""
        parts = [x.strip() for x in filename.rsplit("_", 1)]
        if len(parts) == 1:
            parts = [x.strip() for x in filename.rsplit("-", 1)]
            if len(parts) == 1:
                parts.append("no-publisher")

        title, publisher = parts
        logger.info('Splitting out publisher="%s" title="%s".', publisher, title)
        return title, publisher

    def _make_unique_filename(
        self,
        in_path: Path,
        publish_date: str,
        publisher: str,
        title: str,
    ) -> Path:
        """Return a unique filename by appending a counter."""
        candidate = (
            self._out_path / f"{publish_date}_{publisher}_{title}{in_path.suffix}"
        )
        count = 1
        while candidate.exists():
            candidate = (
                self._out_path
                / f"{publish_date}_{publisher}_{title}_{count}{in_path.suffix}"
            )
            count += 1

        return candidate


def run(input_dir: Path, output_dir: Path, model: str, date_format: str) -> int:
    """Rename each PDF found in the input directory and copy to
    the output directory."""
    if not input_dir.exists():
        return 0

    files = list(input_dir.glob("*.pdf"))
    if not files:
        return 0

    if not output_dir.exists():
        logger.error("Output directory does not exist.")
        return 1

    logger.info("Found (%s) files to modify.", len(files))
    renamer = Renamer(model, date_format, output_dir)
    for count, file in enumerate(files, 1):
        logger.info("Processing file (%s / %s).", count, len(files))
        try:
            renamed = renamer.rename(file)
        except Exception as e:
            logger.exception(e)
        else:
            if renamed is not None:
                file.replace(output_dir / renamed)
    logger.info("Renamed (%s) files.", len(files))

    return 0


if __name__ == "__main__":
    args = parser.parse_args()
    if args.logfile is not None:
        logger.addHandler(logging.FileHandler(args.logfile))
    sys.exit(
        run(Path(args.input_dir), Path(args.output_dir), args.model, args.date_format)
    )
