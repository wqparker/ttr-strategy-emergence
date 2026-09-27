# Development Log

This is a development log to track project progression. 

## 9/22/26 

Initial creation of the repo. Very much a planning stage.

## 9/23/26

Further refining of claude and plan md files. Emphasis on planning things more meticuously now so agents have easier time working on their own with fewer mistakes. Instructued it to run on its own for 5 hours and stepped away for said time, unfortunately it hit what it deemed open-ended design question and stopped about 25 minutes in. Annoying but lesson learned. What work it accomplished on phases 0-2 of roadmap look fine at surface glance, but will take a bit of time to work through it and double check everything. Added ASCII visuals for simlpe visual checks (I do love ASCII stuff). 

## 9/24/26

Testing putting in behavior input changes for Claude itself. Going for tersness, short, and clarity with brevity. Continuing work to expand render view with border tiles to display player positions and fields (caards etc) and other relevant info to game and analysis. Fairly happy with how board is for now, very serviceable, though there is a bit too much blank space and I'd want to fill and/or scale other parts up. Added notes as 'viewer backlog' for smaller tweaks and things, such easy to remedy further in development rather than requireing immediate attention in the moment. Rebasing some parts where I can. 

## 9/25/26

Testing putting in behavior input changes for Claude itself. Going for tersness, short, and clarity with brevity. Continuing work to expand render view with border tiles to display player positions and fields (caards etc) and other relevant info to game and analysis. Fairly happy with how board is for now, very serviceable, though there is a bit too much blank space and I'd want to fill and/or scale other parts up. Added notes as 'viewer backlog' for smaller tweaks and things, such easy to remedy further in development rather than requireing immediate attention in the moment. Rebasing some parts where I can. 

## 9/26/26

Working on post game analysis and stats, gathered per game and per batch of games, full screen scale view. Fixed and touched up deeper potential errors in randomness and gamestate logic, randomness of seeds, randomness of player turn order. Begging to progress to more state machine building with action enocoding etc. I want to do a more in-depth dev report at end with visual states and detailed analysis of how exactly things went - I think I'll just go over commits by date for this. Just passed midnight but have been working on creating and testing first iterations of training linear Q and SARSA agents: want to do initial attempt of what features to use and work out kinks in overall process of agent development. Created intense and arguably over enthusiastic data analysis Matplot page, but data nums make brain go brrrr sometimes. 