from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchCase:
    id: str
    intent: str
    relevant: set[str] = field(default_factory=set)
    anti: set[str] = field(default_factory=set)
    note: str = ""


# ---------------------------------------------------------------------------
# Core benchmark — canonical, well-known answers so hits are reproducible.
# ---------------------------------------------------------------------------
CASES: list[BenchCase] = [
    BenchCase(
        id="ical",
        intent=(
            "a lightweight python library to parse and validate iCalendar (.ics) "
            "files, actively maintained, cpu-only, not a thin wrapper around another tool"
        ),
        relevant={"allenporter/ical", "collective/icalendar", "ics-py/ics-py", "niccokunzmann/python-recurring-ical-events"},
        anti={"pimutils/khal", "Kozea/Radicale", "intrepidcs/python_ics", "allenporter/icaldav"},
        note="library not CLI/server; misleading 'python_ics' is hardware, not calendars",
    ),
    BenchCase(
        id="http-client",
        intent="a modern async-capable python http client library for calling REST APIs",
        relevant={"encode/httpx", "aio-libs/aiohttp", "psf/requests"},
        anti={"httpie/cli", "httpie/httpie"},
        note="library, not the httpie CLI",
    ),
    BenchCase(
        id="vector-db",
        intent="an open-source vector database for similarity search over embeddings",
        relevant={
            "qdrant/qdrant", "milvus-io/milvus", "chroma-core/chroma",
            "weaviate/weaviate", "facebookresearch/faiss", "lancedb/lancedb",
        },
        anti=set(),
        note="dedicated vector DB, not a generic ORM",
    ),
    BenchCase(
        id="pdf-extract",
        intent="a python library to extract text and tables from PDF files",
        relevant={
            "jsvine/pdfplumber", "pymupdf/PyMuPDF", "pdfminer/pdfminer.six",
            "py-pdf/pypdf", "camelot-dev/camelot",
        },
        anti=set(),
        note="text/table extraction, not a PDF viewer app",
    ),
    BenchCase(
        id="data-validation",
        intent="a python library for data validation and settings management using type hints",
        relevant={"pydantic/pydantic", "python-jsonschema/jsonschema", "keleshev/schema"},
        anti=set(),
        note="pydantic is the canonical answer",
    ),
    BenchCase(
        id="cli-framework",
        intent="a python library for building command-line interfaces with argument parsing",
        relevant={"pallets/click", "fastapi/typer", "tiangolo/typer", "google/python-fire", "google/python-fire"},
        anti=set(),
        note="a library to BUILD CLIs, not a specific CLI app",
    ),
    BenchCase(
        id="task-queue",
        intent="a distributed task queue for python to run background jobs",
        relevant={"celery/celery", "rq/rq", "dramatiq/dramatiq"},
        anti=set(),
        note="background job processing",
    ),
    BenchCase(
        id="tui",
        intent="a python framework for building rich text user interfaces in the terminal",
        relevant={"Textualize/textual", "Textualize/rich", "urwid/urwid"},
        anti=set(),
        note="TUI framework",
    ),
    BenchCase(
        id="web-scraping",
        intent="a python framework for large-scale web scraping and crawling",
        relevant={"scrapy/scrapy", "MechanicalSoup/MechanicalSoup"},
        anti=set(),
        note="a scraping framework",
    ),
    BenchCase(
        id="orm",
        intent="a python sql toolkit and object relational mapper for databases",
        relevant={"sqlalchemy/sqlalchemy", "coleifer/peewee", "MagicStack/asyncpg"},
        anti=set(),
        note="an ORM/SQL toolkit",
    ),
]


# ---------------------------------------------------------------------------
# Held-out set — a SEPARATE, unseen benchmark used only for final testing. The
# confidence gate was never tuned against these, and they deliberately span other
# ecosystems (Go, Rust, JS, C++, ML) and phrasings so a good score here proves the
# system generalises rather than overfitting the core cases above.
# ---------------------------------------------------------------------------
HELDOUT_CASES: list[BenchCase] = [
    BenchCase(
        id="go-http-router",
        intent="a fast lightweight http router and web framework for go",
        relevant={
            "gin-gonic/gin", "labstack/echo", "go-chi/chi", "gofiber/fiber",
            "gorilla/mux", "julienschmidt/httprouter", "beego/beego",
        },
        anti=set(),
        note="Go web framework/router",
    ),
    BenchCase(
        id="rust-web",
        intent="an async web framework for rust to build backend http services",
        relevant={
            "tokio-rs/axum", "actix/actix-web", "rwf2/Rocket",
            "SergioBenitez/Rocket", "poem-web/poem", "seanmonstar/warp",
        },
        anti=set(),
        note="Rust async web framework",
    ),
    BenchCase(
        id="js-date",
        intent="a modern lightweight javascript library for parsing and formatting dates",
        relevant={"date-fns/date-fns", "iamkun/dayjs", "moment/luxon"},
        anti={"moment/moment"},
        note="modern+lightweight explicitly excludes legacy heavy moment.js",
    ),
    BenchCase(
        id="react-state",
        intent="a state management library for react applications",
        relevant={
            "reduxjs/redux", "reduxjs/redux-toolkit", "pmndrs/zustand",
            "pmndrs/jotai", "mobxjs/mobx", "facebookexperimental/Recoil",
        },
        anti=set(),
        note="React state management",
    ),
    BenchCase(
        id="rust-serde",
        intent="a rust library for serializing and deserializing data structures to json",
        relevant={"serde-rs/serde", "serde-rs/json"},
        anti=set(),
        note="serde is canonical",
    ),
    BenchCase(
        id="cpp-json",
        intent="a c++ library for parsing and generating json",
        relevant={
            "nlohmann/json", "simdjson/simdjson", "Tencent/rapidjson",
            "open-source-parsers/jsoncpp",
        },
        anti=set(),
        note="C++ JSON library",
    ),
    BenchCase(
        id="go-cli",
        intent="a go library for building command line applications with subcommands and flags",
        relevant={"spf13/cobra", "urfave/cli", "alecthomas/kingpin", "spf13/pflag"},
        anti=set(),
        note="library to BUILD Go CLIs",
    ),
    BenchCase(
        id="ml-tracking",
        intent="a tool for tracking machine learning experiments, metrics and model versions",
        relevant={
            "mlflow/mlflow", "wandb/wandb", "aimhubio/aim",
            "IDSIA/sacred", "Netflix/metaflow", "comet-ml/opik",
        },
        anti=set(),
        note="experiment tracking",
    ),
    BenchCase(
        id="dataframe",
        intent="a fast dataframe library for data analysis in python",
        relevant={
            "pandas-dev/pandas", "pola-rs/polars", "vaexio/vaex",
            "modin-project/modin", "duckdb/duckdb",
        },
        anti=set(),
        note="dataframe/data-analysis library",
    ),
    BenchCase(
        id="graphql-server",
        intent="a library for building a graphql api server in python",
        relevant={
            "graphql-python/graphene", "strawberry-graphql/strawberry",
            "mirumee/ariadne", "graphql-python/gql",
        },
        anti=set(),
        note="Python GraphQL server library",
    ),
    BenchCase(
        id="python-testing",
        intent="a python testing framework with fixtures and simple assert statements",
        relevant={"pytest-dev/pytest", "nose2/nose2", "HypothesisWorks/hypothesis"},
        anti=set(),
        note="pytest is canonical",
    ),
    BenchCase(
        id="message-broker",
        intent="a high performance message broker for publish subscribe and streaming",
        relevant={
            "nats-io/nats-server", "apache/kafka", "apache/pulsar",
            "rabbitmq/rabbitmq-server", "redpanda-data/redpanda", "nsqio/nsq",
        },
        anti=set(),
        note="message broker / streaming platform",
    ),
]


# ---------------------------------------------------------------------------
# Edge / robustness suite — no gold hits; we assert the tool degrades gracefully.
# ---------------------------------------------------------------------------
@dataclass
class EdgeCase:
    id: str
    intent: str
    expectation: str  # human-readable expectation, checked loosely by the runner


EDGE_CASES: list[EdgeCase] = [
    EdgeCase(
        id="impossible",
        intent="a python library for physical time travel to the year 3000",
        expectation="no strong match; must not crash and must not confidently pick junk",
    ),
    EdgeCase(
        id="gibberish",
        intent="asdfghjkl qwerty zxcvbn florb glorp nurgle",
        expectation="no meaningful candidates; graceful empty/weak result",
    ),
    EdgeCase(
        id="one-word",
        intent="parser",
        expectation="ambiguous; must still return a ranked list without error",
    ),
    EdgeCase(
        id="non-english",
        intent="una biblioteca de python para leer y escribir archivos CSV",
        expectation="Spanish for a python CSV read/write library; should find pandas/csv libs",
    ),
    EdgeCase(
        id="adversarial-name",
        intent="a python library named like 'ics' but for parsing calendar files, "
        "NOT hardware or vehicle CAN-bus tools",
        expectation="must reject intrepidcs/python_ics (misleading name) as off_target",
    ),
    EdgeCase(
        id="emoji-noise",
        intent="🚀🔥 a blazing fast json parser for python 🤯 that goes brrr",
        expectation="emoji/hype noise; should still find orjson/ujson/json libs",
    ),
]
