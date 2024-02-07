import streamlit as st
import pandas as pd
import pickle
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Function to send empty emails
def send_empty_emails(emails):
    # Your email credentials and SMTP server information
    sender_email = 'vangalapavan777@gmail.com'
    password = 'jryzcsjexbyhnyov'
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587

    # Connect to the SMTP server
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, password)

        # Send empty emails to each recipient
        for email in emails:
            # Create a new MIME object for each email
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = email
            msg['Subject'] = "Unlock Savings with Our Exclusive Telecom Voucher! "
            # Attach an empty body
            msg.attach(MIMEText("Dear [Recipient],\n\nGreat news! We're excited to share an exclusive telecom voucher with you. Enjoy significant savings on your next telecom bill. Don't miss out on this opportunity to stay connected while saving big.\n\nBest regards,\n[Your Company]", 'plain'))

            server.sendmail(sender_email, email, msg.as_string())
            print(f"Empty email sent successfully to {email}")

def main():
    st.title("TELECOM CUSTOMER CHURN")

    # Load dataset
    df = pd.read_csv('churn-bigml-20.csv')

    df2 = df.copy()
    # Load Indian names
    indian_names_df = pd.read_csv('Indian_Names.csv')
    # Add "Name" column to the dataset
    df2.insert(0, 'Name', indian_names_df['Name'])

    # Generate a list of 666 example email addresses
    emails = [f"example{i}@gmail.com" for i in range(3, 668)]
    emails.insert(0,"pavan.vangala215@gmail.com")
    emails.insert(0,"b20cs158@kitsw.ac.in")
    # Create a DataFrame with the emails
    emails_df = pd.DataFrame(emails, columns=["Email"])

    # Add "Email" column as the second column in df2
    df2.insert(1, 'Email', emails_df['Email'])

    # Display the loaded dataset
    df = df.drop('Churn', axis=1)
    st.write("Loaded Dataset:")
    df2 = df2.drop('Churn', axis=1)
    st.write(df2)

    # Remove 'Area code', 'State', and 'Churn' columns
    columns_to_remove = ['Area code', 'State']
    df_without_columns = df.drop(columns=columns_to_remove, axis=1)

    # Apply one-hot encoding using pd.get_dummies
    df_encoded = pd.get_dummies(df_without_columns)

    # Multi-Select rows from the updated dataset
    all_records_option = "Select All Records"
    selected_rows = st.multiselect("Select Rows:", [all_records_option] + df_encoded.index.tolist())

    # Define result_df outside the "Result" button block
    result_df = pd.DataFrame(columns=['Name', 'Email'])

    # Button to classify selected records and send empty emails
    if st.button("Result"):
        if not selected_rows:
            st.warning("Please select at least one row.")
        else:
            # Check if "Select All Records" is selected
            if all_records_option in selected_rows:
                selected_rows = df_encoded.index

            # Filter the selected rows
            df_selected = df_encoded.loc[selected_rows]

            # Preprocess the data to handle non-numeric values
            df_selected = df_selected.apply(pd.to_numeric, errors='coerce')

            # Drop rows with missing values
            df_selected = df_selected.dropna()

            # Make predictions using the loaded model
            predictions = model.predict(df_selected)

            # Create a DataFrame for 'Churn' rows
            churn_df = df_selected[predictions == 1]

            # Display the 'Churn' DataFrame or "None" if no churn predictions
            if len(churn_df) == 0:
                st.warning("No one is predicted to churn.")
                result_df = pd.DataFrame(columns=['Name', 'Email'])  # Reset to an empty DataFrame
            else:
                st.write("People who are likely to churn are:")
                # Display the names and emails using the df2 DataFrame
                result_df = df2.loc[churn_df.index, ['Name', 'Email']]
                st.table(result_df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]).set_table_styles([{'selector': 'td', 'props': [('text-align', 'center')]}]).set_table_styles([{'selector': 'tr:hover', 'props': [('background-color', '#d2e4f7')]}]))
                # Display the count of people likely to churn
                st.markdown(f"Overall {len(churn_df)} customers are likely to churn")

                # Send empty emails to customers likely to churn
                churn_emails = result_df['Email'].tolist()
                send_empty_emails(churn_emails)
                st.success(f"Emails are sent to {len(churn_emails)} customers who are likely to churn.")

if __name__ == '__main__':
    main()  