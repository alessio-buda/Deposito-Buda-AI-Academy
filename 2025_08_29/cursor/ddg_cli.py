import argparse
import sys

from ddg_client import search_first_text, search_instant_answer, pretty_print


def main(argv=None):
    parser = argparse.ArgumentParser(description="DuckDuckGo Instant Answer CLI")
    parser.add_argument("query", help="Search query string", nargs="+")
    parser.add_argument("--no-html", dest="no_html", action="store_true", help="Request plain text fields")
    parser.add_argument("--allow-disambig", dest="skip_disambig", action="store_false", help="Do not skip disambiguation pages")
    parser.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout in seconds")
    args = parser.parse_args(argv)

    query = " ".join(args.query)
    text = search_first_text(
        query,
        no_html=args.no_html,
        skip_disambig=args.skip_disambig,
        timeout_seconds=args.timeout,
    )
    if text:
        print(text)
    else:
        # Fallback to full JSON for debugging when no concise text is available
        result = search_instant_answer(
            query,
            no_html=args.no_html,
            skip_disambig=args.skip_disambig,
            timeout_seconds=args.timeout,
        )
        print(pretty_print(result))


if __name__ == "__main__":
    raise SystemExit(main())


