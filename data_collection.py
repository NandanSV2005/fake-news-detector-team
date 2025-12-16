# data_collection.py
import pandas as pd
import numpy as np
import os

class DataCollector:
    def __init__(self):
        print("Data Collector Initialized!")
    
    def create_sample_data(self):
        """Create sample true and fake news data for testing"""
        
        # Sample True News
        true_news = [
            "Scientists at Harvard University discovered a new species of amphibian in the Amazon rainforest. The research was published in the journal Nature after five years of study.",
            "The global economy shows signs of recovery as stock markets in the US and Europe reached record highs this quarter, according to financial analysts.",
            "NASA successfully launched the Landsat 9 satellite, designed to monitor climate change effects and provide crucial environmental data for researchers worldwide.",
            "Medical researchers at Oxford University have developed a new vaccine that shows 95% effectiveness in phase 3 clinical trials involving 30,000 participants.",
            "Renewable energy sources now account for 38% of electricity production in Germany, marking a significant milestone in the country's green energy transition.",
            "The Education Ministry announced new digital learning initiatives to improve remote education access for students in rural areas, with a budget of $500 million.",
            "Archaeologists from Cambridge University uncovered an ancient city dating back to 1000 BCE in the Mediterranean region, revealing new insights into early civilization.",
            "A new study published in the British Medical Journal shows that regular exercise reduces the risk of chronic diseases by up to 40% in adults over 50.",
            "The World Health Organization reported a significant decline in malaria cases across Africa, attributing the success to improved mosquito net distribution.",
            "Tech companies announced a joint initiative to develop ethical AI guidelines, focusing on transparency and fairness in algorithmic decision-making.",
            "Climate scientists confirmed that 2023 was the warmest year on record, with global temperatures 1.2°C above pre-industrial levels.",
            "The Federal Reserve maintained interest rates at current levels, citing stable economic growth and controlled inflation metrics.",
            "Researchers at MIT developed a new battery technology that charges electric vehicles in 10 minutes, potentially revolutionizing transportation.",
            "A comprehensive study of ocean currents revealed new patterns that will improve climate models and weather prediction accuracy.",
            "Agricultural scientists developed drought-resistant crop varieties that could help address food security challenges in arid regions.",
            "The United Nations reported progress on Sustainable Development Goals, with 60% of targets showing measurable improvement since 2015."
        ]
        
        # Sample Fake News
        fake_news = [
            "BREAKING: Aliens have been CONFIRMED in Area 51 according to TOP SECRET government documents leaked by an insider! SHOCKING footage reveals EVERYTHING!",
            "URGENT: Drinking diluted bleach cures COVID-19 INSTANTLY, anonymous doctor reveals SECRET treatment that BIG PHARMA doesn't want you to know! SHARE NOW!",
            "5G towers are PROVEN to cause coronavirus outbreaks worldwide! Whistleblower CONFIRMS the connection - this is what THEY don't want you to see!",
            "MOON LANDING was COMPLETELY FAKE and filmed in a Hollywood studio! New EVIDENCE exposes the 50-year conspiracy - NASA has been LYING to everyone!",
            "VACCINES contain MICROCHIPS that allow government to TRACK your movements! Documents PROVE secret surveillance program - WAKE UP people!",
            "Eating just ONE BANANA daily prevents ALL forms of cancer! DOCTORS are HIDING this truth because they profit from chemotherapy!",
            "The Earth is actually FLAT and NASA has spent BILLIONS lying to us! Satellite images are COMPUTER GENERATED - the truth is being SUPPRESSED!",
            "LOSE 10kg in JUST 3 DAYS with this ONE SIMPLE TRICK! Doctors HATE this method because it makes them UNNECESSARY! Click for SECRET!",
            "GOVERNMENT installing MIND CONTROL devices in new smartphones! Update contains secret software that reads your thoughts - PROOF inside!",
            "SECRET cure for aging discovered! 150-year-old scientist reveals how to LIVE FOREVER - pharmaceutical companies trying to SUPPRESS the information!",
            "CHEMTRAILS from airplanes are POISONING our atmosphere! Government ADMITS to weather modification program that's MAKING people sick!",
            "NEW WORLD ORDER plans complete takeover by 2025! Secret documents reveal plot to ELIMINATE cash and implement DIGITAL SLAVERY system!",
            "TIME TRAVEL technology EXISTS and is being used by the elite! Whistleblower reveals how billionaires are manipulating historical events!",
            "HAARP weather weapon CAUSED hurricane that devastated coastal city! Government using climate as a WEAPON against its own citizens!",
            "ALL mainstream media is CONTROLLED by 3 corporations! Every news story is MANUFACTURED to manipulate public opinion - WAKE UP!",
            "CANCER is actually a FUNGUS that can be cured with baking soda! $300 billion medical industry HIDING simple cure to protect profits!"
        ]
        
        # Combine data
        all_texts = true_news + fake_news
        labels = [1] * len(true_news) + [0] * len(fake_news)
        sources = ['Harvard Research', 'Financial Times', 'NASA Official', 'Oxford Study', 
                  'Energy Report', 'Education Ministry', 'Cambridge Archaeology', 'Medical Journal',
                  'WHO Report', 'Tech Consortium', 'Climate Science', 'Federal Reserve', 
                  'MIT Research', 'Oceanography Study', 'Agriculture Dept', 'UN Report'] * 2
        
        # Create DataFrame
        data = {
            'text': all_texts,
            'label': labels,
            'source': sources,
            'text_length': [len(text) for text in all_texts]
        }
        
        df = pd.DataFrame(data)
        
        print(f"Created expanded dataset with {len(df)} articles")
        print(f"True news: {sum(labels)} articles")
        print(f"Fake news: {len(labels) - sum(labels)} articles")
        
        return df
    
    def save_to_csv(self, df, filename='news_data.csv'):
        """Save dataframe to CSV file"""
        if not os.path.exists('data'):
            os.makedirs('data')
        
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to: {filepath}")
        return filepath
    
    def load_from_csv(self, filename='news_data.csv'):
        """Load data from CSV file"""
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"Loaded data from: {filepath}")
            print(f"Shape: {df.shape}")
            return df
        else:
            print(f"File not found: {filepath}")
            return None

def main():
    """Main function to test the data collector"""
    print("=== Fake News Detection System - Data Collection ===")
    
    # Initialize collector
    collector = DataCollector()
    
    # Create sample data
    print("\n1. Creating sample data...")
    df = collector.create_sample_data()
    
    # Display sample data
    print("\n2. Sample data preview:")
    print(df.head())
    
    # Save to CSV
    print("\n3. Saving data to CSV...")
    saved_path = collector.save_to_csv(df)
    
    # Load and verify
    print("\n4. Loading data back to verify...")
    loaded_df = collector.load_from_csv()
    
    if loaded_df is not None:
        print("\n5. Data verification successful!")
        print(f"   Total records: {len(loaded_df)}")
        print(f"   Columns: {list(loaded_df.columns)}")
    
    print("\n=== Data Collection Complete ===")

if __name__ == "__main__":
    main()