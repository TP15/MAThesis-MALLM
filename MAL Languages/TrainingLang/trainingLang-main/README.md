# trainingLang
A simple MAL language used to create simple models and attack graphs for machine learning agents training.

# How to compile trainingLang

1. Download and install malc: https://github.com/mal-lang/malc

2. Clone this repository: `git@github.com:mal-lang/trainingLang.git`

3. Compile trainingLang: `malc trainingLang/src/main/mal/trainingLang.mal`

`malc` will generate a .mar-archive that can be used by MAL Toolbox to generate AttackGraphs.
TODO: Link to docs on how to do that.

# trainingLang
The goal of trainingLang is to be simple but still feature several types of assets and associations.
We want to keep number of attack steps low to result in smaller attack graphs.

# Assets

## Host

Represents a host in a network and can have users and data connected to it.

### Associations 

Host is part of three associations:
- **HostsInNetworks**: Host (hosts) <--> Network (networks)
- **UsersOnHosts**: User (users) <--> Host (hosts)
- **DataOnHosts**: Data (data) <--> Host (hosts)

Note: List above on format `<Asset type> (<field name>) -> <Asset type> (<field name>)`

## Data

Represents data that can be on a host,

Host is part of one association:
- **DataOnHosts**: Data (data) <--> Host (hosts)


## Network

Represents a network that hosts and networks can have access to.

Network is part of two associations:
- **HostsInNetworks**: Host (hosts) <--> Network (networks)
- **InterNetworkConnectivity**:  Network (fromNetworks) <--> Network (toNetworks)


## User

Represents a user that has access to hosts.

Host is part of one association:
- **UsersOnHosts**: User (users) <--> Host (hosts)
