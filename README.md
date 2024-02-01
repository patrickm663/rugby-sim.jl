# A Bayesian approach to simulate the United Rugby Championship

**This project is in very early stages! Right now I am collecting data on the [2022/2023](https://en.wikipedia.org/wiki/2022%E2%80%9323_United_Rugby_Championship) season.**

## Overview
The aim of this project is to develop a model that can estimate the outcome of a rugby game based on the team's 'attack' and 'defence' and a measure of home-ground advantage.

A Bayesian model will be built to estimate suitable ratings per team (much like _FIFA_, but not as detailed as _Rugby 08_). Basically, the model takes in some prior measures for each team, and after observing matches against various teams, updates the measures to form a distribution over, e.g. attack strength. From there, we can simulate possible realities and see how the season plays out.

The current on-going season will be used to test the model with the aim to get it to a point where, following each week's games, the posterior updates via GitHub Actions, ECS, etc.

## License
MIT licensed.
