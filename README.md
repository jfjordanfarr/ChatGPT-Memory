
# ChatGPT Memory Plugin Quickstart

Get the ChatGPT Memory Plugin up and running in under 5 minutes using Python. This plugin gives ChatGPT long-term memory of conversation history to build better context and recall specific exchanges and facts. If you do not already have plugin developer access, please [join the waitlist](https://openai.com/waitlist/plugins).

## Setup

To install the required packages for this plugin, run the following command:

```bash
pip install -r requirements.txt
```

To run the plugin, enter the following command:

```bash
python main.py
```

Once the local server is running:

1. Navigate to https://chat.openai.com. 
2. In the Model drop down, select "Plugins" (note, if you don't see it there, you don't have access yet).
3. Select "Plugin store"
4. Select "Develop your own plugin"
5. Enter in `localhost:5004` since this is the URL the server is running on locally, then select "Find manifest file".

The plugin should now be installed and enabled! You can start by interacting with ChatGPT, and the plugin will store and recall conversation history as needed.

## Getting help

If you run into issues or have questions building a plugin, please join our [Developer community forum](https://community.openai.com/c/chat-plugins/20).
```

This `readme.md` file provides instructions for setting up and running the ChatGPT Memory Plugin. It includes steps for installing the required packages, running the plugin, and installing the plugin in the ChatGPT UI. The plugin runs on the user's local machine from `localhost:5004`. The readme also provides a link to the Developer community forum for assistance and questions.