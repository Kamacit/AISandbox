import os
import tkinter as tk
from tkinter import ttk, simpledialog
from mistralai import Mistral
from exchangelib import Account, Credentials, Configuration, DELEGATE
from exchangelib.protocol import BaseProtocol, NoVerifyHTTPAdapter
from dotenv import load_dotenv
import threading

# Load environment variables from the .env file
load_dotenv()

# Mistral API configuration
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# Function to securely input password via Tkinter
def get_user_credentials():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    email = simpledialog.askstring("Email", "Enter your email address:")
    password = simpledialog.askstring("Password", "Enter your password:", show='*')
    root.destroy()
    return email, password

# Configuration for connecting to Exchange
email, password = get_user_credentials()
credentials = Credentials(email, password)
config = Configuration(server=os.getenv("MAIL_SERVER"), credentials=credentials)
BaseProtocol.HTTP_ADAPTER_CLS = NoVerifyHTTPAdapter

try:
    account = Account(
        primary_smtp_address=email,
        config=config,
        autodiscover=False,
        access_type=DELEGATE
    )
except Exception as e:
    print(f"Error connecting to the Exchange server: {e}")
    exit()

# Fetch only unread emails and sort by urgency
def fetch_emails():
    try:
        emails = account.inbox.filter(is_read=False).order_by('-importance', '-datetime_received')
        return emails
    except Exception as e:
        print(f"Error fetching emails: {e}")
        return []

# Analyze email contents and assign priority and summary
def analyze_emails(emails):
    prioritized_emails = []
    for email in emails:
        try:
            prompt = f"""
            Analyze the following email for urgency and categorize it as 'Low', 'Medium', or 'High' based on the criteria below:

            - **Low Urgency**: The email does not require immediate action and can be addressed at a later time. Examples include newsletters, general updates, or non-time-sensitive information.
            - **Medium Urgency**: The email requires attention within the next few days. Examples include meeting requests, follow-ups, or tasks with a flexible deadline.
            - **High Urgency**: The email requires immediate action or response. Examples include urgent requests, deadlines, or critical issues that need to be addressed promptly.

            Email Subject: {email.subject}
            Email Body: {email.body}

            Based on the content and subject of the email, determine the urgency level.

            Example output: Low

            Example output: Medium

            Example output: High

            """

            chat_response = client.chat.complete(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )

            priority_text = chat_response.choices[0].message.content.strip().lower()
            if "high" in priority_text or "urgent" in priority_text:
                priority_score = "High"
            elif "medium" in priority_text:
                priority_score = "Medium"
            else:
                priority_score = "Low"

            # Analyze the summary
            summary_response = client.chat.complete(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize the following email content in one sentence: {email.body}",
                    },
                ]
            )
            summary = summary_response.choices[0].message.content.strip()

            prioritized_emails.append((email, priority_score, summary))
        except Exception as e:
            print(f"Error analyzing email: {e}")
    return prioritized_emails

# Function to update the GUI
def update_gui(root, tree, emails):
    prioritized_emails = analyze_emails(emails)

    # Sort emails by urgency (High first)
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    prioritized_emails.sort(key=lambda x: priority_order[x[1]])

    for email, priority, summary in prioritized_emails:
        tree.insert('', 'end', values=(email.subject, email.sender.email_address, priority, summary, email.datetime_received.strftime('%Y-%m-%d %H:%M:%S')))

# Create GUI
def create_gui():
    root = tk.Tk()
    root.title("Email Priority List")

    # Progress bar
    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="indeterminate")
    progress.pack(pady=20)

    # Start the progress bar
    progress.start()

    # Treeview for the email list
    tree = ttk.Treeview(root, columns=('Subject', 'Sender', 'Priority', 'Summary', 'Date'), show='headings')
    tree.heading('Subject', text='Subject')
    tree.heading('Sender', text='Sender')
    tree.heading('Priority', text='Priority')
    tree.heading('Summary', text='Summary')
    tree.heading('Date', text='Date')
    tree.pack(fill=tk.BOTH, expand=True)

    # Background task to fetch and analyze emails
    def load_emails():
        emails = fetch_emails()
        update_gui(root, tree, emails)
        progress.stop()
        progress.pack_forget()  # Hide the progress bar

    # Start the background task
    threading.Thread(target=load_emails, daemon=True).start()

    root.mainloop()

# Start application
create_gui()
