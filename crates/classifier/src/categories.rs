//! Exemplar prompts per intent category. The classifier averages the
//! embeddings of the exemplars for each category to derive a prototype
//! vector, then classifies by argmax cosine similarity.
//!
//! Each category has 3–5 exemplars spanning common enterprise phrasings.
//! The `Other` category has no exemplars — it is the zero-vector fallback
//! that wins when no other prototype scores above ~0.

use tokenova_domain::IntentCategory;

pub fn exemplars(category: IntentCategory) -> &'static [&'static str] {
    match category {
        IntentCategory::CodeGeneration => &[
            "Write a Python function that parses JSON",
            "Implement a binary search in TypeScript",
            "Generate a SQL query to join users and orders",
            "Refactor this function to use async/await",
            "Write unit tests for the login handler",
        ],
        IntentCategory::Summarization => &[
            "Summarize this article in three sentences",
            "Give me a TL;DR of the following document",
            "Condense these meeting notes into bullet points",
            "Summarize the key findings of this research paper",
        ],
        IntentCategory::DataExtraction => &[
            "Extract all email addresses from this text as JSON",
            "Parse the following invoice and return a structured object",
            "Pull out every date and amount from this document",
            "Extract named entities from the following passage",
        ],
        IntentCategory::QuestionAnswering => &[
            "What is the capital of France?",
            "How does HTTP/2 multiplexing work?",
            "Who invented the transistor?",
            "When was the Treaty of Versailles signed?",
            "Why is the sky blue?",
        ],
        IntentCategory::DocumentReview => &[
            "Review this contract and flag risky clauses",
            "Proofread this document for grammar and clarity",
            "Audit this policy document for compliance issues",
            "Review this pull request description for completeness",
        ],
        IntentCategory::Translation => &[
            "Translate this paragraph from English to Spanish",
            "Translate the following sentence into French",
            "Render this text in German",
            "Traduce este texto al inglés",
        ],
        IntentCategory::CreativeWriting => &[
            "Write a short story about a robot who learns poetry",
            "Compose a haiku about autumn",
            "Draft a fictional dialogue between two strangers on a train",
            "Write a marketing slogan for a sustainable coffee brand",
        ],
        IntentCategory::ConversationalChat => &[
            "Hey, how are you?",
            "What's up, can we chat for a bit?",
            "Good morning! How was your weekend?",
            "Thanks, that was really helpful.",
        ],
        IntentCategory::ReasoningPlanning => &[
            "Plan a step-by-step strategy to reduce infrastructure costs",
            "Reason through the trade-offs between monolith and microservices",
            "Walk through the pros and cons of this architectural decision",
            "Devise a plan to migrate off the legacy database",
        ],
        // Other is the implicit fallback — no prototype, no exemplars.
        IntentCategory::Other => &[],
    }
}
