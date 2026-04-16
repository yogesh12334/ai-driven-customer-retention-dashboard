import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()

def generate_data(n=300000):
    np.random.seed(42)

    data = []

    for i in range(n):
        age = np.random.randint(18, 70)
        tenure = np.random.randint(1, 72)
        monthly = np.random.uniform(300, 3000)
        usage = np.random.uniform(1, 300)
        tickets = np.random.randint(0, 10)

        # churn probability logic (REALISTIC)
        churn_prob = (
            0.3 * (monthly / 3000) +
            0.3 * (tickets / 10) +
            0.2 * (1 - usage / 300) +
            0.2 * (1 - tenure / 72)
        )

        churn = 1 if np.random.rand() < churn_prob else 0

        data.append([
            i,
            age,
            tenure,
            monthly,
            usage,
            tickets,
            fake.city(),
            np.random.choice(["Monthly", "Yearly"]),
            np.random.choice(["UPI", "Card", "NetBanking"]),
            churn
        ])

    df = pd.DataFrame(data, columns=[
        "CustomerID", "Age", "Tenure", "MonthlyCharges",
        "UsageHours", "SupportTickets", "Location",
        "ContractType", "PaymentMethod", "Churn"
    ])

    df.to_csv("data/customers.csv", index=False)
    print("✅ Data Generated Successfully")

if __name__ == "__main__":
    generate_data()