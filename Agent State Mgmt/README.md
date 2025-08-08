# Stateless vs Stateful

In a stateless system, every prompt is treated independently. LLMs work this way unless given context in the input. In contrast, a stateful system tracks what has already happened, like the items in an online shopping cart. This ability to retain context is essential for agents that take multiple steps or interact with tools across time. When the task is done state is gone unless we chose to persist it. 

A type system like Python’s TypedDict can define the agent's state structure. 

# Memory Is a Simulation

- Ephemeral (In-application): Exists only during the session. Once the interaction ends, the memory disappears. This is useful for short-term context and is the focus of this discussion.
- Durable (Persisted): Stored in databases or vector stores and available across sessions or days. This enables long-term memory but is not covered here.

## Strategies for Simulating Short-Term Memory
- Full Conversation History
- Sliding Window
- Summarization

## State vs. Session Memory
State refers to transient execution information—what the agent is doing right now. It disappears once the task ends.

Session memory covers the broader conversation, tracking tools used, decisions made, and prior responses across steps. While state helps with immediate logic, memory helps the agent act coherently over time.


# Four Dimensions of Evaluation
Effective evaluation looks across multiple dimensions:

- Task Completion: Did the agent achieve its goal? How many steps did it take? Was human intervention needed?
- Quality Control: Was the output in the correct format? Did it follow prompt instructions and use the provided context?
- Tool Interaction: Did the agent choose appropriate tools and use valid arguments?
- System Metrics: How efficient was the run in terms of time, token usage, and failure rates?