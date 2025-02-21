import os
import tkinter as tk
from tkinter import ttk, simpledialog
from mistralai import Mistral
from exchangelib import Account, Credentials, Configuration, DELEGATE
from exchangelib.protocol import BaseProtocol, NoVerifyHTTPAdapter
from dotenv import load_dotenv
import threading

# Laden Sie die Umgebungsvariablen aus der .env-Datei
load_dotenv()

# Mistral-API-Konfiguration
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-large-latest"
client = Mistral(api_key=api_key)

# Funktion zur sicheren Eingabe des Passworts über Tkinter
def get_user_credentials():
    root = tk.Tk()
    root.withdraw()  # Versteckt das Hauptfenster
    email = simpledialog.askstring("E-Mail", "Geben Sie Ihre E-Mail-Adresse ein:")
    password = simpledialog.askstring("Passwort", "Geben Sie Ihr Passwort ein:", show='*')
    root.destroy()
    return email, password

# Konfiguration für die Verbindung zu Exchange
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
    print(f"Fehler beim Verbinden mit dem Exchange-Server: {e}")
    exit()

# Nur ungelesene E-Mails abrufen und nach Dringlichkeit sortieren
def fetch_emails():
    try:
        emails = account.inbox.filter(is_read=False).order_by('-importance', '-datetime_received')
        return emails
    except Exception as e:
        print(f"Fehler beim Abrufen der E-Mails: {e}")
        return []

# E-Mail-Inhalte analysieren und Priorität sowie Zusammenfassung zuweisen
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

            # Analyse der Zusammenfassung
            summary_response = client.chat.complete(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Summarize the following email content in German in one sentence: {email.body}",
                    },
                ]
            )
            summary = summary_response.choices[0].message.content.strip()

            prioritized_emails.append((email, priority_score, summary))
        except Exception as e:
            print(f"Fehler beim Analysieren der E-Mail: {e}")
    return prioritized_emails

# Funktion zum Aktualisieren der GUI
def update_gui(root, tree, emails):
    prioritized_emails = analyze_emails(emails)

    # Sortieren der E-Mails nach Dringlichkeit (High zuerst)
    priority_order = {'High': 0, 'Medium': 1, 'Low': 2}
    prioritized_emails.sort(key=lambda x: priority_order[x[1]])

    for email, priority, summary in prioritized_emails:
        tree.insert('', 'end', values=(email.subject, email.sender.email_address, priority, summary, email.datetime_received.strftime('%Y-%m-%d %H:%M:%S')))

# GUI erstellen
def create_gui():
    root = tk.Tk()
    root.title("E-Mail Prioritätenliste")

    # Ladebalken
    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="indeterminate")
    progress.pack(pady=20)

    # Starten Sie den Ladebalken
    progress.start()

    # Treeview für die E-Mail-Liste
    tree = ttk.Treeview(root, columns=('Subject', 'Sender', 'Priority', 'Summary', 'Date'), show='headings')
    tree.heading('Subject', text='Betreff')
    tree.heading('Sender', text='Absender')
    tree.heading('Priority', text='Priorität')
    tree.heading('Summary', text='Zusammenfassung')
    tree.heading('Date', text='Datum')
    tree.pack(fill=tk.BOTH, expand=True)

    # Hintergrundaufgabe zum Abrufen und Analysieren von E-Mails
    def load_emails():
        emails = fetch_emails()
        update_gui(root, tree, emails)
        progress.stop()
        progress.pack_forget()  # Versteckt den Ladebalken

    # Starten Sie die Hintergrundaufgabe
    threading.Thread(target=load_emails, daemon=True).start()

    root.mainloop()

# Anwendung starten
create_gui()

