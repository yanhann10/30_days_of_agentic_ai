import os
import re
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(__file__))
READING = os.path.join(ROOT, 'reading_progress.md')


def update_read_status(email_body):
    try:
        pattern = r'\b([01])\s+([01])\s+([01])\b'
        match = re.search(pattern, email_body)

        if not match:
            print("No read status pattern found (e.g., '1 0 1')")
            return False

        status_bits = [match.group(1), match.group(2), match.group(3)]
        print(f"Found read status: {' '.join(status_bits)}")

        with open(READING, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        updated = False
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]

            if re.match(r'^\d{4}-\d{2}-\d{2}$', line.strip()):
                date_idx = i

                for idx, bit in enumerate(status_bits, start=1):
                    paper_idx = date_idx + idx
                    if paper_idx < len(lines):
                        paper_line = lines[paper_idx]
                        if paper_line.startswith(f"{idx} | "):
                            parts = paper_line.rsplit(' | ', 1)
                            if len(parts) == 2:
                                lines[paper_idx] = f"{parts[0]} | {bit}\n"
                                updated = True

                break

        if updated:
            with open(READING, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            print(f"✓ Updated read status in {READING}")
            return True
        else:
            print("No papers found to update")
            return False

    except Exception as e:
        print(f"Error updating read status: {e}")
        return False


if __name__ == "__main__":
    test_email = "I read the papers. My status: 1 1 0"
    update_read_status(test_email)
