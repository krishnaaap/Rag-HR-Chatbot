import re
from datetime import datetime, timedelta
from langchain_groq import ChatGroq
from utils_email import format_leave_email
import dateparser
import calendar
import json

# --- Demo Holiday List ---
PUBLIC_HOLIDAYS = [
    datetime(2025, 8, 15),  # Independence Day
    datetime(2025, 8, 28),  # Example holiday
    datetime(2025, 9, 5)    # Example holiday
]

# 🔒 Removed sensitive API key → now use environment variable
import os
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-70b-8192", temperature=0.3)

# -------------------- Date Parsing Helpers -------------------- #
def parse_date_safe(date_str, base_date=None):
    """Parse a date string safely, removing ordinal suffixes."""
    if base_date is None:
        base_date = datetime.now()
    date_str = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str, flags=re.IGNORECASE)
    try:
        return dateparser.parse(date_str, settings={"RELATIVE_BASE": base_date})
    except Exception:
        return None

def filter_leave_dates(dates: list, public_holidays: list[datetime]):
    """Separate valid working days from weekends, holidays, or past dates."""
    valid_dates, invalid_dates = [], []
    today = datetime.today().date()

    for d in dates:
        dt_date = d.date() if isinstance(d, datetime) else d
        if isinstance(d, str):
            dt = parse_date_safe(d)
            if dt:
                dt_date = dt.date()
            else:
                continue
        if dt_date < today or dt_date.weekday() >= 5 or dt_date in [h.date() for h in public_holidays]:
            invalid_dates.append(dt_date)
        else:
            valid_dates.append(dt_date)

    return sorted(set(valid_dates)), sorted(set(invalid_dates))

def weekday_name_to_date(query, today=None):
    if today is None:
        today = datetime.today().date()
    weekdays = list(calendar.day_name)
    for i, day_name in enumerate(weekdays):
        if day_name.lower() in query.lower():
            days_ahead = (i - today.weekday() + 7) % 7
            if days_ahead == 0:
                days_ahead = 7
            return [(today + timedelta(days=days_ahead)).strftime("%d %B %Y")]
    return []

# -------------------- Leave Extraction -------------------- #
def extract_leave_dates(query: str):
    now = datetime.now()
    dates = set()

    # Handle ranges e.g., "21–25 August"
    range_match = re.search(r'(\d{1,2}(?:st|nd|rd|th)?\s*\w*\s*\d{0,4})\s*(?:to|[-–])\s*(\d{1,2}(?:st|nd|rd|th)?\s*\w*\s*\d{0,4})', query, re.IGNORECASE)
    if range_match:
        start_str, end_str = range_match.groups()
        start_date = parse_date_safe(start_str, now)
        end_date = parse_date_safe(end_str, now)
        if start_date and end_date:
            cur = start_date
            while cur <= end_date:
                dates.add(cur.date())
                cur += timedelta(days=1)

    # Month-specific days e.g., "22nd and 25th August"
    month_match = re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s*\d{0,4}', query, re.IGNORECASE)
    if month_match:
        month_str = month_match.group(0)
        prefix = query[:month_match.start()]
        suffix = query[month_match.start():]
        for day in re.findall(r'(\d{1,2})(?:st|nd|rd|th)?', prefix + suffix):
            d = parse_date_safe(f"{day} {month_str}", now)
            if d:
                dates.add(d.date())

    # Fully-specified dates e.g., "25 August"
    all_dates = re.findall(r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*(?:\s+\d{4})?)\b', query, re.IGNORECASE)
    for date_str in all_dates:
        d = parse_date_safe(date_str.strip(), now)
        if d:
            dates.add(d.date())

    # Weekdays fallback
    weekday_dates = weekday_name_to_date(query, now.date())
    for date_str in weekday_dates:
        d = parse_date_safe(date_str, now)
        if d:
            dates.add(d.date())

    # Relative days fallback (today, tomorrow, etc.)
    for rp in re.findall(r'\b(today|tomorrow|day after tomorrow|next\s+week|this\s+week)\b', query, re.IGNORECASE):
        d = parse_date_safe(rp, now)
        if d:
            dates.add(d.date())

    return sorted(dates)

# -------------------- Safety & Validation -------------------- #
def detect_toxicity(text, toxicity_classifier):
    result = toxicity_classifier(text)[0]
    return result["label"].lower() == "toxic" and result["score"] > 0.6

def is_valid_date(date_str: str) -> bool:
    return dateparser.parse(date_str) is not None

def validate_date_range(dates: list[str]) -> list[str]:
    valid_dates = [d for d in dates if is_valid_date(d)]
    return sorted(valid_dates, key=lambda x: dateparser.parse(x))

def format_leave_dates(dates: list[str]) -> str:
    if not dates:
        return ""
    parsed_dates = sorted([datetime.strptime(d, "%d %B %Y") for d in dates])
    ranges, start, end = [], parsed_dates[0], parsed_dates[0]
    for dt in parsed_dates[1:]:
        if (dt - end).days == 1:
            end = dt
        else:
            ranges.append(f"{start.strftime('%d %B %Y')}" if start == end else f"{start.strftime('%d %B %Y')} to {end.strftime('%d %B %Y')}")
            start = end = dt
    ranges.append(f"{start.strftime('%d %B %Y')}" if start == end else f"{start.strftime('%d %B %Y')} to {end.strftime('%d %B %Y')}")
    return "on " + ", ".join(ranges)
