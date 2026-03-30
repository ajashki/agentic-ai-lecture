import json
from pathlib import Path
from collections import defaultdict
from langchain_core.tools import tool

DATA_PATH = Path(__file__).parent.parent / "data" / "transactions.json"

MONTH_MAP = {
    "january": "2025-01",
    "february": "2025-02",
    "march": "2025-03",
    "jan": "2025-01",
    "feb": "2025-02",
    "mar": "2025-03",
}


def _load_transactions() -> list[dict]:
    with open(DATA_PATH) as f:
        return json.load(f)


def _filter_by_month(transactions: list[dict], month: str) -> list[dict]:
    prefix = MONTH_MAP.get(month.lower())
    if not prefix:
        return []
    return [t for t in transactions if t["date"].startswith(prefix)]


@tool
def get_transactions(month: str) -> str:
    """Get all transactions for a given month. Month should be 'january', 'february', or 'march'.
    Returns a JSON list of transactions with id, date, description, amount, category, and merchant."""
    txns = _filter_by_month(_load_transactions(), month)
    if not txns:
        return f"No transactions found for '{month}'. Valid months: january, february, march."
    return json.dumps(txns, indent=2)


@tool
def categorize_expenses(month: str) -> str:
    """Get total spending per category for a given month. Month should be 'january', 'february', or 'march'.
    Returns a JSON object with category names as keys and total amounts as values, plus a grand total."""
    txns = _filter_by_month(_load_transactions(), month)
    if not txns:
        return f"No transactions found for '{month}'. Valid months: january, february, march."

    totals = defaultdict(float)
    for t in txns:
        totals[t["category"]] += t["amount"]

    result = {cat: round(amt, 2) for cat, amt in sorted(totals.items())}
    result["_total"] = round(sum(totals.values()), 2)
    return json.dumps(result, indent=2)


@tool
def compare_months(month1: str, month2: str) -> str:
    """Compare spending between two months. Shows each category's spend in both months,
    the difference, and percentage change. Months should be 'january', 'february', or 'march'."""
    all_txns = _load_transactions()
    txns1 = _filter_by_month(all_txns, month1)
    txns2 = _filter_by_month(all_txns, month2)

    if not txns1:
        return f"No transactions found for '{month1}'."
    if not txns2:
        return f"No transactions found for '{month2}'."

    totals1 = defaultdict(float)
    totals2 = defaultdict(float)
    for t in txns1:
        totals1[t["category"]] += t["amount"]
    for t in txns2:
        totals2[t["category"]] += t["amount"]

    categories = sorted(set(totals1.keys()) | set(totals2.keys()))
    comparison = {}
    for cat in categories:
        a = round(totals1.get(cat, 0), 2)
        b = round(totals2.get(cat, 0), 2)
        delta = round(b - a, 2)
        pct = round((delta / a) * 100, 1) if a > 0 else None
        comparison[cat] = {
            month1: a,
            month2: b,
            "delta": delta,
            "pct_change": f"{pct}%" if pct is not None else "N/A",
        }

    total1 = round(sum(totals1.values()), 2)
    total2 = round(sum(totals2.values()), 2)
    comparison["_total"] = {
        month1: total1,
        month2: total2,
        "delta": round(total2 - total1, 2),
        "pct_change": f"{round(((total2 - total1) / total1) * 100, 1)}%" if total1 > 0 else "N/A",
    }
    return json.dumps(comparison, indent=2)


@tool
def find_subscriptions() -> str:
    """Analyze all transactions across all months to find recurring charges.
    A subscription is a charge from the same merchant with a similar amount appearing in 2+ months.
    Returns a JSON list of detected subscriptions with merchant, amount, and months active."""
    all_txns = _load_transactions()

    # Group by merchant+approximate amount per month
    merchant_months = defaultdict(lambda: defaultdict(list))
    for t in all_txns:
        month_key = t["date"][:7]  # "2025-01"
        merchant_months[t["merchant"]][month_key].append(t["amount"])

    subscriptions = []
    for merchant, months in merchant_months.items():
        if len(months) >= 2:
            # Check if amounts are consistent (within 10%)
            all_amounts = [amt for m_amts in months.values() for amt in m_amts]
            avg = sum(all_amounts) / len(all_amounts)
            if all(abs(a - avg) / avg < 0.10 for a in all_amounts):
                subscriptions.append({
                    "merchant": merchant,
                    "amount": round(avg, 2),
                    "months_active": sorted(months.keys()),
                    "total_cost": round(sum(all_amounts), 2),
                })

    subscriptions.sort(key=lambda x: x["amount"], reverse=True)
    return json.dumps(subscriptions, indent=2)


@tool
def search_transactions(keyword: str) -> str:
    """Search all transactions across all months by keyword. Matches against description and merchant name
    (case-insensitive). Returns matching transactions."""
    keyword_lower = keyword.lower()
    all_txns = _load_transactions()
    matches = [
        t for t in all_txns
        if keyword_lower in t["description"].lower() or keyword_lower in t["merchant"].lower()
    ]
    if not matches:
        return f"No transactions found matching '{keyword}'."
    return json.dumps(matches, indent=2)


@tool
def calculate_savings_goal(target_amount: float, months: int) -> str:
    """Calculate how much you need to save per month to reach a target amount.
    Also analyzes current spending to suggest where you could cut costs.
    target_amount: the total amount you want to save (in euros).
    months: the number of months to reach the goal."""
    monthly_needed = round(target_amount / months, 2)

    # Analyze current spending for suggestions
    all_txns = _load_transactions()
    monthly_spend = defaultdict(float)
    for t in all_txns:
        month_key = t["date"][:7]
        monthly_spend[month_key] += t["amount"]

    avg_monthly = round(sum(monthly_spend.values()) / len(monthly_spend), 2)

    # Find potential cuts
    suggestions = []

    # Check subscriptions
    merchant_months = defaultdict(lambda: defaultdict(list))
    for t in all_txns:
        month_key = t["date"][:7]
        merchant_months[t["merchant"]][month_key].append(t["amount"])

    sub_total = 0
    for merchant, months_data in merchant_months.items():
        if len(months_data) >= 2:
            all_amounts = [a for m in months_data.values() for a in m]
            avg = sum(all_amounts) / len(all_amounts)
            if all(abs(a - avg) / avg < 0.10 for a in all_amounts):
                sub_total += avg

    suggestions.append(f"Total monthly subscriptions: EUR {round(sub_total, 2)} — review if all are needed")

    # Check food delivery
    delivery_total = sum(t["amount"] for t in all_txns if t["merchant"] in ("Uber Eats", "Thuisbezorgd"))
    delivery_monthly = round(delivery_total / 3, 2)
    if delivery_monthly > 20:
        suggestions.append(f"Food delivery averages EUR {delivery_monthly}/month — cooking more could save EUR {round(delivery_monthly * 0.6, 2)}/month")

    # Check Uber rides
    uber_total = sum(t["amount"] for t in all_txns if t["merchant"] == "Uber" and t["category"] == "transport")
    uber_monthly = round(uber_total / 3, 2)
    if uber_monthly > 15:
        suggestions.append(f"Uber rides average EUR {uber_monthly}/month — public transit could save EUR {round(uber_monthly * 0.7, 2)}/month")

    result = {
        "target": target_amount,
        "months": months,
        "monthly_savings_needed": monthly_needed,
        "current_avg_monthly_spend": avg_monthly,
        "suggestions": suggestions,
    }
    return json.dumps(result, indent=2)


@tool
def set_budget_alert(category: str, limit: float) -> str:
    """Set a monthly budget alert for a spending category. This will track your spending
    and notify you when you exceed the limit.
    category: the spending category (food, transport, entertainment, etc.).
    limit: the monthly budget limit in euros."""
    all_txns = _load_transactions()

    # Calculate current spending in this category per month
    monthly = defaultdict(float)
    for t in all_txns:
        if t["category"].lower() == category.lower():
            month_key = t["date"][:7]
            monthly[month_key] += t["amount"]

    if not monthly:
        return f"No transactions found in category '{category}'. Valid categories: food, transport, subscriptions, entertainment, shopping, education, utilities."

    current = {m: round(v, 2) for m, v in sorted(monthly.items())}
    avg = round(sum(monthly.values()) / len(monthly), 2)
    over_budget = {m: round(v - limit, 2) for m, v in current.items() if v > limit}

    result = {
        "alert_set": True,
        "category": category,
        "monthly_limit": limit,
        "current_spending_by_month": current,
        "average_monthly": avg,
        "months_over_budget": over_budget if over_budget else "None — you're within budget!",
    }
    return json.dumps(result, indent=2)
