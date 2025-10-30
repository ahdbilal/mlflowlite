# Enrich and Process Data with AI

This guide demonstrates how to use Databricks AI Functions to enrich and process your data with artificial intelligence. AI Functions are built-in SQL functions that apply AI capabilities like sentiment analysis, text generation, and document parsing directly to your data.

We'll explore three approaches, progressing from simple to advanced use cases.

## Prerequisites

- Access to Databricks SQL or notebooks
- Databricks Runtime 15.4 LTS or above
- Access to Foundation Model APIs (see [region availability](https://docs.databricks.com/aws/en/resources/feature-region-support))

## Sample Dataset

For these examples, we'll use the `samples.bakehouse.media_customer_reviews` table, which contains customer reviews for a bakery. You can use any table with text data in your environment.

---

## Task-Based AI Functions

Task-based AI Functions are the easiest way to get started. They provide high-level AI capabilities for common tasks without requiring any model configuration.

### Example: Analyze Sentiment

Understand how customers feel about your products:

```sql
SELECT 
  review,
  ai_analyze_sentiment(review) AS sentiment
FROM samples.bakehouse.media_customer_reviews
LIMIT 10;
```

**Output:** Returns `positive`, `negative`, `neutral`, or `mixed`.

### Other Available Task-Based Functions

- **`ai_summarize(text)`** - Generate summaries
- **`ai_classify(text, labels)`** - Classify into categories
- **`ai_extract(text, entities)`** - Extract specific information
- **`ai_translate(text, target_language)`** - Translate text
- **`ai_fix_grammar(text)`** - Correct grammatical errors
- **`ai_mask(text, entities)`** - Mask sensitive information (PII)

📖 [See all task-specific functions](https://docs.databricks.com/aws/en/large-language-models/ai-functions#task-specific-ai-functions)

**When to use:** Perfect for common AI tasks and getting started quickly.

---

## General-Purpose AI with `ai_query`

The `ai_query()` function gives you full control to apply any AI model with custom prompts. This is ideal when you need specific outputs or want to tune the AI's behavior.

### Example: Generate Custom Action Items

Create actionable insights from customer reviews:

```sql
SELECT 
  review,
  ai_query(
    "databricks-gpt-oss-120b",
    CONCAT("Generate 3 specific action items for the business based on this customer review: ", review)
  ) AS action_items
FROM samples.bakehouse.media_customer_reviews
LIMIT 10;
```

### Advanced Usage

Control the model's behavior with parameters:

```sql
SELECT 
  review,
  ai_query(
    "databricks-gpt-oss-120b",
    CONCAT("Write a professional response to this customer review: ", review),
    modelParameters => named_struct('max_tokens', 200, 'temperature', 0.7)
  ) AS response_draft
FROM samples.bakehouse.media_customer_reviews
LIMIT 10;
```

### Recommended Models

📖 [See all available models and advanced parameters](https://docs.databricks.com/aws/en/large-language-models/ai-functions#ai_query)

**When to use:** Use `ai_query` when you need custom prompting or specific output formats.

---

## Process Unstructured Documents with `ai_parse_document`

The `ai_parse_document()` function extracts structured content from unstructured documents like PDFs, images, and office documents. This is powerful for document processing pipelines.

### Setup: Store Documents in Unity Catalog Volumes

First, upload your documents to a Unity Catalog Volume:

```sql
-- Create a volume for storing documents
CREATE VOLUME IF NOT EXISTS main.default.customer_documents;
```

Upload documents to `/Volumes/main/default/customer_documents/` using the Databricks UI or CLI.

### Example: Parse PDF Documents

Extract text and structure from PDF files:

```sql
SELECT 
  path,
  ai_parse_document(
    path,
    returnType => 'markdown'
  ) AS parsed_content
FROM (
  SELECT 
    'dbfs:/Volumes/main/default/customer_documents/feedback_form_001.pdf' AS path
);
```

### Combine with `ai_query` for Analysis

Process documents and analyze the content:

```sql
WITH parsed_docs AS (
  SELECT 
    filename,
    ai_parse_document(
      CONCAT('dbfs:/Volumes/main/default/customer_documents/', filename),
      returnType => 'markdown'
    ) AS content
  FROM (
    SELECT EXPLODE(ARRAY(
      'customer_feedback_1.pdf',
      'customer_feedback_2.pdf'
    )) AS filename
  )
)
SELECT 
  filename,
  ai_query(
    'databricks-gpt-oss-120b',
    CONCAT('Summarize the key points from this document: ', content)
  ) AS summary
FROM parsed_docs;
```

### Supported File Formats

- **Documents:** PDF
- **Images:** JPG, JPEG, PNG

📖 [See advanced parsing options and return types](https://docs.databricks.com/aws/en/sql/language-manual/functions/ai_parse_document)

**When to use:** Use `ai_parse_document` for extracting text from documents, processing scanned images, or automating data entry from forms.

---

## Using AI Functions in Python

AI Functions work in PySpark too:

```python
df = spark.table("samples.bakehouse.media_customer_reviews")

df_enriched = df.selectExpr(
    "review",
    "ai_analyze_sentiment(review) as sentiment",
    "ai_summarize(review) as summary"
)

df_enriched.write.mode("overwrite").saveAsTable("enriched_reviews")
```

---

## Best Practices

1. **Start Simple:** Begin with task-based functions before moving to `ai_query`
2. **Test Small:** Use `LIMIT` to test on small datasets first to manage costs
3. **Batch Processing:** Use Databricks Workflows for large-scale processing
4. **Monitor Costs:** Check usage in `system.billing.usage` table

---

## Next Steps

- 📚 [Full AI Functions Documentation](https://docs.databricks.com/aws/en/large-language-models/ai-functions)
- 🚀 [Deploy Batch Inference Pipelines](https://docs.databricks.com/aws/en/machine-learning/model-serving/batch-inference)
- 📊 [Monitor Model Serving Costs](https://docs.databricks.com/aws/en/machine-learning/model-serving/monitor-costs)

---

## Quick Reference

| Task | Function | Level |
|------|----------|-------|
| Sentiment Analysis | `ai_analyze_sentiment(text)` | 1 |
| Summarization | `ai_summarize(text)` | 1 |
| Classification | `ai_classify(text, labels)` | 1 |
| Custom Prompts | `ai_query(model, prompt)` | 2 |
| Document Parsing | `ai_parse_document(path)` | 3 |

**Ready to get started?** Try the Level 1 examples first with your own data!

