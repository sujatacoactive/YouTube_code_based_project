import os
import argparse
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LLM setup
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0.7
)

# Helper function to save task outputs
def save_output(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# Define Agents
researcher_agent = Agent(
    role="YouTube Researcher",
    goal="Research video ideas around a given keyword.",
    backstory="Expert in identifying trending topics on YouTube.",
    allow_delegation=False,
    llm=llm
)

scriptwriter_agent = Agent(
    role="Script Writer",
    goal="Write a clear and engaging script for a YouTube video based on a given keyword.",
    backstory="Creative storyteller and video script writer.",
    allow_delegation=False,
    llm=llm
)

seo_agent = Agent(
    role="SEO Specialist",
    goal="Optimize titles, description, tags, and hashtags strictly based on the given keyword.",
    backstory="SEO expert for YouTube channels.",
    allow_delegation=False,
    llm=llm
)

thumbnail_agent = Agent(
    role="Thumbnail Designer",
    goal="Generate a thumbnail design prompt with overlay text for the given keyword.",
    backstory="Expert in designing eye-catching YouTube thumbnails.",
    allow_delegation=False,
    llm=llm
)

# Define Tasks
def create_tasks(keyword: str):
    research_task = Task(
        description=(
            "Research 5 YouTube video ideas around the keyword: '{keyword}'. "
            "Pick the most promising idea. Ensure ideas are directly related to '{keyword}'."
        ),
        agent=researcher_agent,
        inputs={"keyword": keyword},
        expected_output="out/research.md"
    )

    script_task = Task(
        description=(
            "Write a ~220 word engaging YouTube script for the topic strictly based on the keyword: '{keyword}'. "
            "The script should be conversational and informative, avoid drifting off-topic."
        ),
        agent=scriptwriter_agent,
        inputs={"keyword": keyword},
        expected_output="out/script.md"
    )

    seo_task = Task(
        description=(
            "Based ONLY on the keyword '{keyword}' and the outputs of research + script, "
            "generate YouTube SEO metadata. Return JSON with 3 click-worthy titles (must contain the keyword), "
            "a 200-300 word description (keyword-rich), 10-12 tags (keyword-based), 3 hashtags, and a suggested schedule."
        ),
        agent=seo_agent,
        inputs={"keyword": keyword},
        expected_output="out/metadata.json"
    )

    thumbnail_task = Task(
        description=(
            "Generate 3 thumbnail prompt ideas strictly for the keyword '{keyword}'. "
            "Each should include a visual concept and suggested overlay text containing the keyword."
        ),
        agent=thumbnail_agent,
        inputs={"keyword": keyword},
        expected_output="out/thumbnail_prompt.txt"
    )

    return [research_task, script_task, seo_task, thumbnail_task]

# Crew setup
def run_pipeline(keyword: str):
    tasks = create_tasks(keyword)

    crew = Crew(
        agents=[researcher_agent, scriptwriter_agent, seo_agent, thumbnail_agent],
        tasks=tasks,
        process=Process.sequential
    )

    result = crew.kickoff(inputs={"keyword": keyword})

    # Save outputs
    for task in tasks:
        if task.output:
            save_output(task.expected_output, str(task.output))
            print(f"✅ Saved {task.expected_output}")

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--keyword", type=str, required=True, help="Keyword for YouTube content generation")
    args = parser.parse_args()

    run_pipeline(args.keyword)
