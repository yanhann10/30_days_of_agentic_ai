from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


def generate_improved_document(
    original_doc: str,
    recommendations: list[str],
    output_path: str
) -> str:
    agent = ChatAnthropic(model="claude-sonnet-4-20250514", max_tokens=8192)

    prompt = f"""You are tasked with improving a research document based on expert debate recommendations.

=== ORIGINAL DOCUMENT ===
{original_doc}

=== MODERATOR RECOMMENDATIONS FROM DEBATES ===
{chr(10).join(recommendations)}

=== YOUR TASK ===
1. Integrate the accepted recommendations into the document
2. Add new sections as recommended
3. Enhance existing sections with the proposed content
4. Maintain the document's structure and style
5. Update the "Last updated" date to today

Output the complete improved document in markdown format.
Preserve all existing valuable content while adding the improvements.
"""

    messages = [
        SystemMessage(content="You are an expert technical writer specializing in AI/ML research documentation."),
        HumanMessage(content=prompt)
    ]

    response = agent.invoke(messages)
    improved_doc = response.content

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(improved_doc)

    print(f"Improved document saved to: {output_path}")
    return improved_doc
