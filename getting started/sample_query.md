SELECT 
  review,
  ai_query("databricks-claude-sonnet-4", CONCAT("Generate action items for us based on this review: ", review)) as action_items 
FROM samples.bakehouse.media_customer_reviews 
LIMIT 1
