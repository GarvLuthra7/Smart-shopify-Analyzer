import numpy as np
import pandas as pd

np.random.seed(42)

product_type_map={
    "Essential":1.0,
    "Productivity":0.8,
    "Lifestyle":0.5,
    "Luxury":0.2
}

product_types=list(product_type_map.keys())

def generate_data(n=2000):
    data=[]
    for _ in range(n):
        salary=np.random.randint(20000,200000)
        expenses=np.random.randint(5000,salary-5000)
        price=np.random.randint(500,100000)
        usage=np.random.randint(1,30)

        q_answers=np.random.randint(0,2,5)
        need_score=np.mean(q_answers)

        product_type=np.random.choice(product_types)
        product_type_score=product_type_map[product_type]

        final_need_score=(need_score*0.7) + (product_type_score*0.3)

        disposable_income=salary-expenses

        safety=(0.4*(disposable_income/salary)+0.3*(usage/30)+(0.3*final_need_score))

        safety=max(0,min(1,safety))
        safety-=(price/salary)*0.3

        regret=1-safety+np.random.normal(0,0.05)
        regret=max(0,min(1,regret))

        data.append([price,salary,expenses, usage, final_need_score,safety,regret])
    
    df=pd.DataFrame(data,columns=["price","salary","expenses", "usage", "need_score","safety","regret"])

    return df

if __name__=="__main__":
    df=generate_data()
    df.to_csv("shopping_data.csv", index=False)
    print("Updated dataset generated!")