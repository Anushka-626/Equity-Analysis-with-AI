# pip install llama-cpp-python
from llama_cpp import Llama, LlamaGrammar
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('reuters.db')
cursor = conn.cursor()

# Add the 'final_score' column if it doesn't exist
try:
    cursor.execute("""
        ALTER TABLE news ADD COLUMN final_score INTEGER;
    """)
except sqlite3.OperationalError:
    # Ignore error if the column already exists
    pass


# Fetch all news articles
cursor.execute("SELECT news_id, headline, text FROM news")
news_articles = cursor.fetchall()

# Prompt template
prompt_template = """
News Article:
Headline: {headline}

Text: {text}

Based on the above news article, provide a yes/no - answer for each of these points.

Topic: Financial Market Relevance
1. Does the news affect major stock indices (e.g., S&P 500, FTSE 100, Nikkei 225)?
2. Does the news involve a major global economic policy change (e.g., interest rates, trade policies)?
3. Is the news about a significant geopolitical event (e.g., war, major diplomatic agreements)?
4. Does the news impact major global currencies (e.g., USD, EUR, JPY)?
5. Is the news related to significant changes in commodity prices (e.g., oil, gold)?
6. Does the news concern major corporations or industries (e.g., tech giants, automotive industry)?
7. Is there an impact on major global financial institutions (e.g., banks, hedge funds)?
8. Does the news affect international trade or supply chains?
9. Is the news related to significant economic data releases (e.g., GDP, unemployment rates)?
10. Does the news lead to significant market speculation or volatility?

Respond in this way:
    - Provide your reasoning for each question.
    - THEN provide a final answer for the yes/no question.

"""

root_rule = 'root ::= ' + ' '.join([f'"Reasoning {i} (1 sentence max): " reasoning "\\n" "Answer {i} (yes/no): " yesno "\\n"' for i in range(1, 11)])

grammar = f"""
{root_rule}
reasoning ::= sentence
sentence ::= [^.!?]+[.!?]
yesno ::= "yes" | "no"
"""

grammar = LlamaGrammar.from_string(grammar)

# Put the full model path here! Example: model_path = r"C:\\models\\Llama-3-Instruct-8B-SPPO-Iter3-Q5_K_M.gguf",
# My recommendation: Llama-3-Instruct-8B-SPPO-Iter3-Q5_K_M.gguf (huggingface.co!)
print ("Remember to change the model_path variable!")
llm = Llama(
    model_path = r"C:\Users\krati\OneDrive\Desktop\Lectures\Equity Analysis with AI\llama-3-instruct-8B-sppo-iter3-q6_k.gguf",
    n_gpu_layers = -1,
    seed = 123,
    n_ctx = 8192,
    verbose = False
)

for news_id, headline, text in news_articles:
    prompt = prompt_template.format(headline=headline, text=text)
    
    output = llm(
        prompt,
        max_tokens = 2000,
        stop = ["</s>"],
        echo = False,
        grammar = grammar
    )

    response = output["choices"][0]["text"].strip()
    yes_count = sum(1 for line in response.split('\n') if line.startswith("Answer") and line.split(": ")[1].strip().lower() == "yes")
    no_count = sum(1 for line in response.split('\n') if line.startswith("Answer") and line.split(": ")[1].strip().lower() == "no")
    
    final_score = yes_count

    print(f"Headline: {headline}")
    print(f"Response:\n{response}")
    print(f"Yes answers: {yes_count}")
    print(f"No answers: {no_count}")
    print(f"Final Score: {yes_count}")
    print("-" * 50)
    
    # Update the final_score in the database
    cursor.execute("""
        UPDATE news
        SET final_score = ?
        WHERE news_id = ?
    """, (final_score, news_id))
    # Commit the transaction
    conn.commit()
    

# Close the database connection
conn.close()
