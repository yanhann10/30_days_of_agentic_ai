import os
import re
import csv
from datetime import datetime, timedelta
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(__file__))
READING = os.path.join(ROOT, 'reading_progress.md')
READ_RATE_CSV = os.path.join(ROOT, 'read_rate_tracking.csv')
ALERT_FILE = os.path.join(ROOT, 'read_rate_alert.txt')


def extract_read_status_history():
    with open(READING, 'r', encoding='utf-8') as f:
        content = f.read()

    date_pattern = r'^(\d{4}-\d{2}-\d{2})$'
    status_pattern = r'^\d+\s+\|\s+.+?\s+\|\s+([01])$'

    history = []
    current_date = None
    current_statuses = []

    for line in content.split('\n'):
        line = line.strip()

        date_match = re.match(date_pattern, line)
        if date_match:
            if current_date and current_statuses:
                history.append({
                    'date': current_date,
                    'statuses': current_statuses.copy()
                })
            current_date = date_match.group(1)
            current_statuses = []
            continue

        status_match = re.match(status_pattern, line)
        if status_match and current_date:
            current_statuses.append(int(status_match.group(1)))

    if current_date and current_statuses:
        history.append({
            'date': current_date,
            'statuses': current_statuses
        })

    return history


def calculate_read_rate(statuses):
    if not statuses:
        return 0.0
    return sum(statuses) / len(statuses) * 100


def calculate_rolling_read_rate(history, weeks=3):
    if len(history) < weeks:
        return None

    recent_weeks = history[-weeks:]
    all_statuses = []
    for entry in recent_weeks:
        all_statuses.extend(entry['statuses'])

    return calculate_read_rate(all_statuses)


def update_tracking_csv(history):
    file_exists = os.path.exists(READ_RATE_CSV)

    with open(READ_RATE_CSV, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(['date', 'read_status', 'weekly_read_rate', 'rolling_3wk_read_rate'])

        existing_dates = set()
        if file_exists:
            with open(READ_RATE_CSV, 'r', encoding='utf-8') as rf:
                reader = csv.DictReader(rf)
                existing_dates = {row['date'] for row in reader}

        for entry in history:
            if entry['date'] in existing_dates:
                continue

            date = entry['date']
            statuses = entry['statuses']
            read_status_str = ' '.join(str(s) for s in statuses)
            weekly_rate = calculate_read_rate(statuses)

            idx = history.index(entry)
            if idx >= 2:
                rolling_rate = calculate_rolling_read_rate(history[:idx+1], weeks=3)
            else:
                rolling_rate = None

            rolling_rate_str = f"{rolling_rate:.1f}" if rolling_rate is not None else ""

            writer.writerow([date, read_status_str, f"{weekly_rate:.1f}", rolling_rate_str])


def check_for_dip(history):
    if len(history) < 6:
        return False, None, None

    current_3wk = calculate_rolling_read_rate(history[-3:], weeks=3)
    previous_3wk = calculate_rolling_read_rate(history[-6:-3], weeks=3)

    if current_3wk is None or previous_3wk is None:
        return False, None, None

    if current_3wk < previous_3wk:
        return True, previous_3wk, current_3wk

    return False, previous_3wk, current_3wk


def create_alert(previous_rate, current_rate):
    alert_date = datetime.now().strftime('%Y-%m-%d')
    expires = (datetime.now() + timedelta(weeks=1)).strftime('%Y-%m-%d')

    with open(ALERT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"⚠️  READ RATE DIP DETECTED\n")
        f.write(f"Alert created: {alert_date}\n")
        f.write(f"Expires: {expires}\n\n")
        f.write(f"Previous 3-week read rate: {previous_rate:.1f}%\n")
        f.write(f"Current 3-week read rate: {current_rate:.1f}%\n")
        f.write(f"Drop: {previous_rate - current_rate:.1f} percentage points\n\n")
        f.write(f"Action needed: Analyze unread papers and improve selection/digest quality.\n")


def clear_expired_alert():
    if not os.path.exists(ALERT_FILE):
        return

    with open(ALERT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('Expires:'):
            expires_str = line.split(':', 1)[1].strip()
            expires_date = datetime.strptime(expires_str, '%Y-%m-%d')

            if datetime.now() > expires_date:
                os.remove(ALERT_FILE)
                print(f"✓ Cleared expired alert from {expires_str}")
            break


def main():
    print("Tracking read rate...")

    clear_expired_alert()

    history = extract_read_status_history()

    if not history:
        print("No read status data found in reading_progress.md")
        return

    print(f"Found {len(history)} weeks of data")

    update_tracking_csv(history)
    print(f"✓ Updated {READ_RATE_CSV}")

    has_dip, prev_rate, curr_rate = check_for_dip(history)

    if has_dip:
        create_alert(prev_rate, curr_rate)
        print(f"⚠️  READ RATE DIP: {prev_rate:.1f}% → {curr_rate:.1f}%")
        print(f"   Alert created: {ALERT_FILE}")
    else:
        if prev_rate is not None and curr_rate is not None:
            print(f"✓ Read rate stable: {prev_rate:.1f}% → {curr_rate:.1f}%")
        else:
            print("Not enough data for 3-week comparison (need 6+ weeks)")


if __name__ == "__main__":
    main()
