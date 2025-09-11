# claude-in-docker
empty project for a quick setup of Claude in Docker+VSCode

Motivation + bonus step-by-step github authorization setting here: [timsh.org](https://timsh.org/claude-inside-docker/)

Now with docker inside enabled (docker in docker), so it cannot see the images of the host machine.
Todo: sharing host claude settings, a container rebuild should not require a new login

# Prerequisites: 
- Docker installed
- VSCode installed
- Claude somehow paid for

# Steps
- `git clone this repo
(in fact you are better of forking it, until I figure out how templates work)
- `cd claude-in-docker`
- `code` (or open the folder manually in VSCode)
- a popup in the rear right corner should appear offering to "Reopen in Container" - do it!
- wait for a bit...
- in the automatically opened terminal tab you'll see that you're inside docker
- type `claude` to activate Claude Code

You're all set! 
