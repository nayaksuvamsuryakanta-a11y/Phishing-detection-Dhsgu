"""
Synthetic email corpus generator for DHSGSU PhishGuard.

Produces data/emails.csv with a balanced set of legitimate and phishing
messages built by combinatorial explosion of templates, filler paragraphs and
random variations. No real email is copied or scraped: every row is assembled
at runtime from short reusable fragments, so a 2000+ row dataset is generated
from roughly two hundred lines of source data.

Run with:  python data/generate_dataset.py
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

# Resolve paths relative to the project root (the parent of this file's
# package) so the script works from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "emails.csv"

import sys as _sys

_sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    FREE_MAIL_DOMAINS,
    OFFICIAL_SENDERS,
    OFFICIAL_URL,
    SEED,
    SQUAT_DOMAINS,
    THREAT_INTEL_BLOCKLIST,
    UNIVERSITY_SHORT,
)

random.seed(SEED)

# ---------------------------------------------------------------------------
# Fragment vocabulary
# ---------------------------------------------------------------------------

STUDENT_IDS = ["DHSGSU/2023/BT/0482", "DHSGSU/2022/MSC/1156",
               "DHSGSU/2021/BA/0039", "DHSGSU/2024/BCA/0721",
               "DHSGSU/2023/BSC/0544", "DHSGSU/2022/BED/0287",
               "DHSGSU/2024/MA/0093", "DHSGSU/2023/LLB/0162"]

COURSES = ["Bachelor of Computer Applications", "Bachelor of Science",
           "Bachelor of Arts", "Master of Science in Chemistry",
           "Bachelor of Education", "Master of Arts in Sociology",
           "Bachelor of Laws", "Bachelor of Commerce"]

DEPARTMENTS = ["Computer Science", "Chemistry", "Mathematics", "Physics",
               "English", "Commerce", "Education", "Law", "Zoology", "Botany"]

EXAMS = ["Theory Examination", "Practical Examination", "Internal Assessment",
         "Semester Examination", "Supplementary Examination",
         "Viva-Voce Examination"]

MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# ---------------------------------------------------------------------------
# Legit templates
# ---------------------------------------------------------------------------

# Each legit template is a callable(subject, sender_name, sender_email, body, url).
# The tuple form is (subject_fragments, sender_key, body_fragments, url_pool).

LEGIT_SUBJECTS = [
    ["{course} Semester {sem} Exam Timetable Released",
     "Exam Schedule for {course} Semester {sem} - {month} {year}",
     "Revised Time Table: {course} Sem {sem} ({month})",
     "{dept} Department - Examination Date Sheet {year}"],
    ["{course} Semester {sem} Results Declared",
     "Result Notification - {course} Sem {sem} ({month} {year})",
     "Provisional Result List: {course} Semester {sem}",
     "{dept} Department Result Update - {year}"],
    ["Fee Receipt for Academic Year {year}-{next_year}",
     "Fee Payment Confirmation - Receipt No. {receipt}",
     "Your Fee Receipt - {course} Sem {sem}",
     "Receipt of Fees Paid - {month} {year}"],
    ["NAAC Accreditation Notice - {dept} Department",
     "NAAC Peer Team Visit Schedule - {month} {year}",
     "Preparation Meeting for NAAC Accreditation",
     "NAAC Documentation Submission Deadline - {dept}"],
    ["Admission Merit List {year} - {course}",
     "Counselling Schedule for {course} Admissions",
     "Admission Confirmation - Document Verification Date",
     "Seat Allotment Notice - {course} Semester {sem}"],
    ["Convocation {year} Ceremony Notification",
     "Invitation: Annual Convocation - {month} {year}",
     "Degree Distribution Schedule - Convocation {year}",
     "Convocation Registration Opens - {month}"],
    ["Library Notice - Due Date Reminder",
     "Book Renewal Notice - Library Services",
     "Library Card Renewal for {year}",
     "Overdue Book Reminder - Return by {date}"],
    # Real university IT communications legitimately discuss passwords --
    # "change your password every 90 days" is a standard policy notice. A
    # keyword detector that flags every mention of "password" would wrongly
    # reject these. Including them forces the model to learn the difference
    # between an advisory notice and a credential-harvesting lure, which is
    # the hardest distinction in this domain.
    ["IT Security Notice: Password Policy Update",
     "Mandatory Password Reset - {month} {year}",
     "New Password Policy Effective from {date}"],
    ["Scholarship Application Notice {year}",
     "Merit Cum Means Scholarship - Last Date {date}",
     "Scholarship Disbursement Update",
     "Renewal of Scholarship - Submit Form by {date}"],
]

LEGIT_BODY_TEMPLATES = [
    ("Dear Student,\n\n"
     "This is to inform you that the {exam} for {course} Semester {sem} "
     "has been scheduled to begin on {date}. All candidates are requested "
     "to download the detailed timetable from the university examination "
     "portal and carry their admit card to the examination hall.\n\n"
     "Kindly note that no candidate will be permitted without a valid "
     "identity card.\n\n"
     "Regards,\n"
     "Controller of Examinations\n"
     "{university}"),
    ("Dear Student,\n\n"
     "The results for {course} Semester {sem} ({month} {year}) have been "
     "declared. You may now view your result through the official "
     "student portal using your enrollment credentials.\n\n"
     "In case of any discrepancy in the result, please submit a "
     "revaluation application within seven working days.\n\n"
     "Regards,\n"
     "Controller of Examinations\n"
     "{university}"),
    ("Dear Student,\n\n"
     "This is to acknowledge receipt of your fee payment for the academic "
     "year {year}-{next_year}. Your official receipt number is "
     "{receipt}. A soft copy of the receipt is available for download "
     "from the fee portal.\n\n"
     "Please preserve this receipt for future reference.\n\n"
     "Regards,\n"
     "Finance Section\n"
     "{university}"),
    ("Dear Faculty and Student,\n\n"
     "The NAAC peer team visit for the {dept} department is scheduled "
     "from {date_1} to {date_2}. All departments are requested to "
     "complete the documentation and keep the records ready for inspection.\n\n"
     "Your cooperation is solicited for the success of this accreditation "
     "process.\n\n"
     "Regards,\n"
     "NAAC Steering Committee\n"
     "{university}"),
    ("Dear Applicant,\n\n"
     "The merit list for {course} admissions ({year}) has been published "
     "on the university website. Shortlisted candidates must appear for "
     "document verification on {date} along with original certificates.\n\n"
     "Failure to appear will result in cancellation of the allotted seat.\n\n"
     "Regards,\n"
     "Admission Cell\n"
     "{university}"),
    ("Dear Student,\n\n"
     "You are cordially invited to attend the Annual Convocation on "
     "{date} at the university auditorium. Degree certificates will be "
     "distributed by the Chief Guest.\n\n"
     "Kindly register online through the convocation portal before "
     "{deadline}.\n\n"
     "Regards,\n"
     "Registrar\n"
     "{university}"),
    ("Dear Student,\n\n"
     "This is a reminder that the book(s) issued under library card "
     "{receipt} are due for return on {date}. Please return or renew "
     "them before the due date to avoid overdue charges.\n\n"
     "Regards,\n"
     "University Library\n"
     "{university}"),
    ("Dear Student,\n\n"
     "Applications for the {scholarship} scholarship ({year}) are now "
     "invited. Eligible students must submit the completed form along "
     "with supporting documents by {deadline}.\n\n"
     "Late applications will not be entertained under any circumstance.\n\n"
     "Regards,\n"
     "Scholarship Section\n"
     "{university}"),
    ("Dear Student,\n\n"
     "In accordance with the revised information security policy, all "
     "student portal passwords must be changed once every ninety days. "
     "Your current password will remain valid until {date}, and you will "
     "be asked to set a new password on your next login.\n\n"
     "Please choose a strong password combining letters, numbers and "
     "special characters. Remember, the university will never ask you to "
     "send your password or OTP through email.\n\n"
     "Regards,\n"
     "IT Support Desk\n"
     "{university}"),
]

# ---------------------------------------------------------------------------
# Phish templates
# ---------------------------------------------------------------------------

PHISH_SUBJECTS = [
    ["URGENT: Verify Your DHSGSU Portal Account Immediately",
     "Action Required - DHSGSU Student Portal Verification",
     "Verify Your University Account Before Deactivation",
     "DHSGSU Portal: Complete Verification Now",
     "Alert: Your DHSGSU Login Needs Verification"],
    ["Fee Refund Eligibility - Claim Your Amount Now",
     "You Are Eligible For A Fee Refund - Apply Within 24 Hours",
     "DHSGSU Fee Refund Notification - Claim Immediately",
     "Refund Alert: Unclaimed Amount In Your Name",
     "Process Your DHSGSU Fee Refund Today"],
    ["OTP Verification Required For Your Account",
     "Security Alert: OTP Verification Pending",
     "Verify Your Identity - Enter OTP To Continue",
     "DHSGSU Account Security Update - OTP Required",
     "Immediate OTP Authentication Needed"],
    ["Scholarship Disbursement - Update Your Bank Details",
     "Scholarship Amount Pending - Submit Account Details",
     "Your Scholarship Will Be Released After Verification",
     "Urgent: Scholarship Payment Blocked - Update Details",
     "Scholarship Disbursement - Action Required"],
    ["Exam Form Rejection - Rectify Immediately",
     "Your Exam Form Has Been Rejected - Re-Submit Now",
     "Critical: Exam Form Deficiency Notice",
     "Exam Application Invalid - Complete Verification",
     "Notice: Exam Form Submission Incomplete"],
    ["Password Expiry Notice - Reset Your DHSGSU Password",
     "Your Account Password Will Expire In 24 Hours",
     "Security Update: Reset Your Portal Password Now",
     "DHSGSU Password Policy Change - Immediate Action",
     "Account Lockout Warning - Reset Password At Once"],
    ["Certificate Verification - Download Now",
     "Your Degree Certificate Is Ready For Download",
     "Certificate Collection Authorization - Click To Verify",
     "Download Your DHSGSU Certificate Copy",
     "E-Certificate Issuance - Verify And Download"],
    ["Wi-Fi Account Activation Required",
     "DHSGSU Campus Wi-Fi - Activate Your Credentials",
     "Network Access Approval - Login Details Inside",
     "Activate Your University Internet Access Now",
     "Campus Wi-Fi Credentials Pending Verification"],
]

PHISH_BODY_TEMPLATES = [
    ("Dear Student,\n\n"
     "Our security system has detected unusual login activity on your "
     "{university} student portal account. To prevent permanent "
     "deactivation, you must verify your account within the next "
     "24 hours.\n\n"
     "Click the link below to verify your credentials: {url}\n\n"
     "Failure to verify will result in suspension of your portal "
     "access and examination services.\n\n"
     "Regards,\n"
     "IT Support Desk\n"
     "{university}"),
    ("Dear Student,\n\n"
     "You are eligible for a fee refund of Rs. {amount} against the "
     "academic year {year}-{next_year}. The unclaimed amount will be "
     "forfeited if not processed within 24 hours.\n\n"
     "To claim your refund, please verify your bank account details here: "
     "{url}\n\n"
     "Note: Refunds will not be processed after the deadline.\n\n"
     "Regards,\n"
     "Finance Section\n"
     "{university}"),
    ("Dear Student,\n\n"
     "A mandatory security update requires all students to authenticate "
     "their identity. Please provide your one-time password (OTP) to "
     "complete the verification process.\n\n"
     "Enter your OTP at: {url}\n\n"
     "This step is compulsory to avoid blocking your student account.\n\n"
     "Regards,\n"
     "Security Administration\n"
     "{university}"),
    ("Dear Student,\n\n"
     "Your {scholarship} scholarship disbursement has been held due to "
     "incomplete bank account information. Kindly update your account "
     "number and IFSC code immediately so the amount can be released.\n\n"
     "Update your details here: {url}\n\n"
     "Disbursement will be cancelled if the details are not updated "
     "within 48 hours.\n\n"
     "Regards,\n"
     "Scholarship Section\n"
     "{university}"),
    ("Dear Student,\n\n"
     "Your examination form for {course} Semester {sem} has been rejected "
     "due to incomplete information. You must re-submit the form with "
     "corrected details within 24 hours, else you will not be permitted "
     "to appear for the {exam}.\n\n"
     "Re-submit your form at: {url}\n\n"
     "Regards,\n"
     "Controller of Examinations\n"
     "{university}"),
    ("Dear Student,\n\n"
     "As per the updated security policy, your {university} portal "
     "password will expire in 24 hours. You are advised to reset your "
     "password immediately to avoid account lockout.\n\n"
     "Reset your password here: {url}\n\n"
     "Failure to reset will block access to results and exam services.\n\n"
     "Regards,\n"
     "IT Support Desk\n"
     "{university}"),
    ("Dear Student,\n\n"
     "Your degree certificate for {course} has been issued and is ready "
     "for download. Please verify your identity to download the "
     "e-certificate copy.\n\n"
     "Download your certificate: {url}\n\n"
     "The download link will expire in 12 hours.\n\n"
     "Regards,\n"
     "Registrar\n"
     "{university}"),
    ("Dear Student,\n\n"
     "Your {university} campus Wi-Fi account requires activation. "
     "Please login with your student credentials to activate network "
     "access before {date}.\n\n"
     "Activate your account: {url}\n\n"
     "Regards,\n"
     "Computer Centre\n"
     "{university}"),
]

# Subtle phish templates. Deliberately free of urgency, threat and
# credential keywords: the only tell is the look-alike sender domain and the
# off-host link. These exist so the classifier cannot reduce phishing to
# "contains the word OTP".
PHISH_SUBJECTS_SUBTLE = [
    "{course} Semester {sem} Result - View Online",
    "Latest Update Regarding Your {course} Admission",
    "{dept} Department Notice - {month} {year}",
    "Scholarship List {year} - Check Your Name",
    "Your {university} Portal Message Is Awaiting",
]

PHISH_BODY_SUBTLE_TEMPLATES = [
    ("Dear Student,\n\n"
     "The {course} semester {sem} result update has been published on the "
     "student information portal. You may review your record and download "
     "the provisional statement from the link below.\n\n"
     "View details: {url}\n\n"
     "Regards,\n"
     "Academic Section\n"
     "{university}"),
    ("Dear Student,\n\n"
     "A revised circular for the {dept} department has been issued for "
     "{month} {year}. Kindly read the attached notification and confirm "
     "your participation through the link given below.\n\n"
     "Read circular: {url}\n\n"
     "Regards,\n"
     "Head of Department\n"
     "{university}"),
    ("Dear Student,\n\n"
     "The provisional merit list for {scholarship} scholarships ({year}) "
     "is now available. Candidates whose names appear are requested to "
     "follow the instructions in the linked document.\n\n"
     "Open document: {url}\n\n"
     "Regards,\n"
     "Scholarship Section\n"
     "{university}"),
    ("Dear Student,\n\n"
     "Your admission file for {course} requires one additional document "
     "to be uploaded. Please submit the scanned copy using the link below "
     "so that your record can be completed.\n\n"
     "Upload document: {url}\n\n"
     "Regards,\n"
     "Admission Cell\n"
     "{university}"),
    ("Dear Student,\n\n"
     "An administrative update concerning your {university} student "
     "record has been posted. We advise you to verify the information at "
     "the earliest.\n\n"
     "Verify here: {url}\n\n"
     "Regards,\n"
     "Registrar Office\n"
     "{university}"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pick(seq: list) -> str:
    return random.choice(seq)


def _fill(template: str, **subs: str) -> str:
    """Fill a template, silently dropping keys the template does not use."""
    return template.format(**{k: v for k, v in subs.items()
                              if "{" + k + "}" in template})


def _random_date(month: str | None = None) -> str:
    day = random.randint(1, 28)
    month = month or _pick(MONTHS)
    year = random.choice([2024, 2025])
    return f"{day} {month} {year}"


def _random_url(domain: str) -> str:
    paths = ["portal", "login", "verify", "account", "student", "fee",
             "result", "exam", "update", "auth", "claim", "download"]
    return f"https://{domain}/{_pick(paths)}?id={random.randint(1000, 9999)}"


def _official_sender() -> tuple[str, str, str]:
    email = random.choice(list(OFFICIAL_SENDERS))
    return OFFICIAL_SENDERS[email], email, email


def _squat_sender() -> tuple[str, str, str]:
    """Return a display-name + email that mimics an official role."""
    role = random.choice(list(OFFICIAL_SENDERS.values()))
    domain = random.choice(SQUAT_DOMAINS)
    email = f"{role.lower().replace(' ', '.')}@{domain}"
    return role, email, email


def _free_mail_sender() -> tuple[str, str, str]:
    role = random.choice(list(OFFICIAL_SENDERS.values()))
    domain = random.choice(FREE_MAIL_DOMAINS)
    local = f"{role.lower().replace(' ', '.')}{random.randint(1, 999)}"
    email = f"{local}@{domain}"
    return role, email, email


def _ip_url() -> str:
    octet = lambda: str(random.randint(1, 254))
    path = random.choice(["login", "verify", "portal", "secure", "auth"])
    return f"http://{octet()}.{octet()}.{octet()}.{octet()}/{path}"


def _has_blocklisted_url(url: str) -> bool:
    lowered = url.lower()
    return any(indicator in lowered for indicator in THREAT_INTEL_BLOCKLIST)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def generate_legit(count: int) -> list[dict]:
    """Generate legitimate university emails, all from @dhsgsu.edu.in.

    Real university mail is not perfectly benign in its wording: deadline
    notices genuinely use urgency ("final notice", "immediate action"), and
    administrative circulars often carry several links. Injecting these
    patterns deliberately creates *class overlap* so the model cannot cheat
    by memorising a single deterministic rule such as "any mail containing
    'urgent' is phishing". Without this overlap the dataset would be
    trivially separable and the reported metrics would be meaningless.
    """
    rows = []
    for i in range(count):
        sub_template = _pick(LEGIT_SUBJECTS)
        body_template = _pick(LEGIT_BODY_TEMPLATES)
        sender_name, sender_email, _ = _official_sender()

        course = _pick(COURSES)
        dept = _pick(DEPARTMENTS)
        sem = random.choice(["I", "II", "III", "IV", "V", "VI"])
        exam = _pick(EXAMS)
        month = _pick(MONTHS)
        year = str(random.choice([2023, 2024, 2025]))
        next_year = str(int(year) + 1)
        receipt = f"DHSGSU/R/{random.randint(10000, 99999)}"
        scholarship = random.choice(["Merit-cum-Means", "National",
                                     "State Government", "Post-Matric"])

        subject = _fill(
            _pick(sub_template), course=course, sem=sem, dept=dept,
            exam=exam, month=month, year=year, next_year=next_year,
            receipt=receipt, date=_random_date(month),
            date_1=_random_date(month), date_2=_random_date(month),
            deadline=_random_date(month), university=UNIVERSITY_SHORT,
        )

        body = _fill(
            body_template, course=course, sem=sem, dept=dept, exam=exam,
            month=month, year=year, next_year=next_year, receipt=receipt,
            date=_random_date(month), date_1=_random_date(month),
            date_2=_random_date(month), deadline=_random_date(month),
            scholarship=scholarship, amount=str(random.randint(5000, 25000)),
            university=UNIVERSITY_SHORT,
        )

        # Deliberate noise: a subset of official mail carries urgency
        # language, multiple links, or terse wording -- all patterns that
        # also appear in phishing mail.
        if random.random() < 0.28:
            urgency = _pick([
                " -- action required by the deadline indicated above.",
                " Kindly treat this as urgent and act immediately.",
                " Final notice: non-compliance will attract a penalty.",
                " Immediate attention of all concerned is solicited.",
            ])
            body = body.rstrip() + "\n\n" + urgency

        urls = []
        if random.random() < 0.70:
            urls.append(
                f"{OFFICIAL_URL}{random.choice(['exam', 'result', 'fee', 'notice'])}"
            )
        if random.random() < 0.22:
            urls.append(
                f"{OFFICIAL_URL}{random.choice(['portal', 'library', 'scholarship', 'login'])}"
            )
        url = "  ".join(urls)

        # A short official alert (SMS-style) keeps the body-length feature
        # from being a clean divider between the two classes.
        if random.random() < 0.08:
            body = (
                f"Dear Student, {subject}. "
                f"See {OFFICIAL_URL}notice for details. Regards, University."
            )

        rows.append({
            "subject": subject,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "body": body,
            "url": url,
            "spf_pass": random.random() < 0.97,
            "dkim_pass": random.random() < 0.97,
            "dmarc_pass": random.random() < 0.95,
            "has_blocklisted_url": _has_blocklisted_url(url),
            "label": 0,
        })
    return rows


def generate_phish(count: int) -> list[dict]:
    """Generate phishing emails using squat domains, free mail and IP URLs.

    Two difficulty tiers are produced. 'overt' phish carry the classic
    giveaways (credential asks, urgency, IP links); 'subtle' phish rely only
    on a look-alike domain and a soft lure such as "view your result here" --
    exactly the pattern that fools readers who trust a plausible sender
    address. Mixing the tiers stops 'asks_for_password_or_otp' and
    'has_ip_url' from becoming deterministic labels, which is what made the
    corpus trivially separable in the first pass.
    """
    rows = []
    for i in range(count):
        tier = "subtle" if random.random() < 0.32 else "overt"

        if tier == "subtle":
            subject = _pick(PHISH_SUBJECTS_SUBTLE)
            body_template = _pick(PHISH_BODY_SUBTLE_TEMPLATES)
        else:
            subject = _pick(_pick(PHISH_SUBJECTS))
            body_template = _pick(PHISH_BODY_TEMPLATES)

        # Rotate sender spoofing technique so the model learns multiple
        # attack surfaces: look-alike domains, free-mail impersonation,
        # and raw IP-address links.
        tactic = random.choices(
            ["squat", "freemail", "ip"], weights=[0.45, 0.35, 0.20]
        )[0]
        # Subtle phish never use raw IP links -- those are a giveaway that
        # would defeat the purpose of the subtle tier.
        if tier == "subtle" and tactic == "ip":
            tactic = "squat"
        if tactic == "squat":
            sender_name, sender_email, _ = _squat_sender()
            domain_pool = SQUAT_DOMAINS + THREAT_INTEL_BLOCKLIST[:4]
            url = _random_url(random.choice(domain_pool))
        elif tactic == "freemail":
            sender_name, sender_email, _ = _free_mail_sender()
            url = _random_url(random.choice(FREE_MAIL_DOMAINS))
        else:
            sender_name, sender_email, _ = _squat_sender()
            url = _ip_url()

        course = _pick(COURSES)
        dept = _pick(DEPARTMENTS)
        sem = random.choice(["I", "II", "III", "IV", "V", "VI"])
        exam = _pick(EXAMS)
        month = _pick(MONTHS)
        year = str(random.choice([2024, 2025]))
        next_year = str(int(year) + 1)
        scholarship = random.choice(["Merit-cum-Means", "National",
                                     "State Government", "Post-Matric"])

        subject = _fill(
            subject, course=course, sem=sem, dept=dept, exam=exam,
            month=month, year=year, university=UNIVERSITY_SHORT,
        )

        # The URL appears in the body AND as the 'url' field, which is how a real
# email's link count is measured. Varying the number of additional decoy
# links keeps 'url_count' from being a deterministic label (previously every
# phish had exactly 2 links, which made the corpus trivially separable).
        decoys = ""
        if random.random() < 0.28:
            decoys = "  " + _random_url(random.choice(SQUAT_DOMAINS))
        body = _fill(
            body_template, course=course, sem=sem, dept=dept, exam=exam,
            month=month, year=year, next_year=next_year,
            date=_random_date(month), scholarship=scholarship,
            amount=str(random.randint(5000, 25000)),
            university=UNIVERSITY_SHORT, url=url,
        ) + decoys

        rows.append({
            "subject": subject,
            "sender_name": sender_name,
            "sender_email": sender_email,
            "body": body,
            "url": url,
            "spf_pass": random.random() < 0.12,
            "dkim_pass": random.random() < 0.12,
            "dmarc_pass": random.random() < 0.10,
            "has_blocklisted_url": _has_blocklisted_url(url),
            "label": 1,
        })
    return rows


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings; used as a heuristic feature."""
    a, b = a.lower(), b.lower()
    if len(a) < len(b):
        a, b = b, a
    if b == "":
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1,          # deletion
                           cur[j - 1] + 1,       # insertion
                           prev[j - 1] + (ca != cb)))  # substitution
        prev = cur
    return prev[-1]


def build_dataset(legit_count: int = 1000, phish_count: int = 1000,
                  shuffle: bool = True) -> list[dict]:
    rows = generate_legit(legit_count) + generate_phish(phish_count)
    if shuffle:
        random.shuffle(rows)
    return rows


def save_dataset(rows: list[dict], path: Path | str = DATA_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject", "sender_name", "sender_email", "body", "url",
        "spf_pass", "dkim_pass", "dmarc_pass", "has_blocklisted_url", "label",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    random.seed(SEED)
    rows = build_dataset()
    saved = save_dataset(rows)
    legit = sum(1 for r in rows if r["label"] == 0)
    phish = sum(1 for r in rows if r["label"] == 1)
    print(f"Wrote {len(rows)} rows to {saved}")
    print(f"  legit (label=0): {legit}")
    print(f"  phish  (label=1): {phish}")
    # Quick sanity audit of the attack-surface distribution in the phish half.
    squat = sum(1 for r in rows if r["label"] == 1
                and any(d in r["sender_email"] for d in SQUAT_DOMAINS))
    free = sum(1 for r in rows if r["label"] == 1
               and any(d in r["sender_email"] for d in FREE_MAIL_DOMAINS))
    ip = sum(1 for r in rows if r["label"] == 1
             and re.match(r"^https?://\d+\.\d+\.\d+\.\d+", r["url"]))
    print(f"  phish by sender type — squat domains: {squat}, "
          f"free-mail: {free}, raw IP URL: {ip}")


if __name__ == "__main__":
    main()