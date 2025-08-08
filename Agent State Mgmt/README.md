# Stateless vs Stateful

In a stateless system, every prompt is treated independently. LLMs work this way unless given context in the input. In contrast, a stateful system tracks what has already happened, like the items in an online shopping cart. This ability to retain context is essential for agents that take multiple steps or interact with tools across time.

When the task is done state is gone unless we chose to persist it. 