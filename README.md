# Climate DT (DE_340) workflow

Welcome to the Climate DT Workflow documentation!

## Getting Started

All the experiments should be created in the [Autosubmit Virtual Machine](https://wiki.eduuni.fi/display/cscRDIcollaboration/Autosubmit+VM). To access the VM, you need a user and your SSH key to be authorized. Add your name, e-mail, preferred username, and SSH (local) public key to the [table](https://wiki.eduuni.fi/display/cscRDIcollaboration/Autosubmit+VM+Users).

Make sure you have a recent Autosubmit version running `autosubmit --version`. Check by doing `module spider autosubmit` and then selecting the latest version of Autosubmit. The version of Autosubmit that has been used for each release can also be found in the [CHANGELOG.md](CHANGELOG.md). You can follow more detailed description about Autosubmit in [Autosubmit Readthedocs](https://autosubmit.readthedocs.io/en/master/).

### Prerequisites

Inside the Autosubmit VM, you need to put your user configurations for platforms somewhere (we recommend `~/platforms.yml`):

```
# personal platforms file
# this overrides keys the default platforms.yml

Platforms:
  lumi-login:
    USER: <USER>
  lumi:
    USER: <USER>
  marenostrum5:
    USER: <USER>
  marenostrum5-login:
    USER: <USER>
```

You also need to configure password-less access to the platforms where you want to run
experiments. Further instructions can be found [here](https://wiki.eduuni.fi/display/cscRDIcollaboration/Autosubmit+VM) (Section 4. How to get password-less access from VM to LUMI / MN5).
The workflow can run as a `local` project or as a `git` project.

### Create your own experiment

1. Create an Autosubmit experiment using minimal configurations.

> **NOTE**: you MUST change `<TYPE_YOUR_PLATFORM_HERE>` below with your platform, and add a description.
For example: lumi or marenostrum5
> Check the available platforms at `/appl/AS/DefaultConfigs/platforms.yml`
> or `~/platforms.yml` if you created this file in your home directory.

```
autosubmit expid \
  --description "A useful description" \
  --HPC <TYPE_YOUR_PLATFORM_HERE> \
  --minimal_configuration \
  --git_as_conf conf/bootstrap/ \
  --git_repo https://earth.bsc.es/gitlab/digital-twins/de_340-2/workflow \
  --git_branch <latest minor tag>
```

> **NOTE:** Do not forget to change the keys between `< >`

You will receive the following message: `Experiment <expid> created`, where `<expid>`
is the identifier of your experiment. A directory will be created for your experiment
at: `/appl/AS/AUTOSUBMIT_DATA/<expid>`.

Basic instructions to execute the workflow with Autosubmit can be found in [How to run](https://earth.bsc.es/gitlab/digital-twins/de_340-2/workflow/-/wikis/How-to-run). The rest of the documentation is available in the same [Wiki](https://earth.bsc.es/gitlab/digital-twins/de_340-2/workflow/-/wikis/home).

2. Start the interactive command line to setup the main characteristics of your experiment: write(models), read(apps,simless) or streaming(end-to-end); simulation mode: historical, projection, control

```
# type anywhere in the virtual machine
autosubmit create <expid>
```

> **NOTE:** You are asked to introduce the credentials to the remote repositories, therefore you must have access to it to proceed.

You will be further asked to select the type of workflow you want to run:

```
Choose a workflow:
1. model
2. end_to_end
3. app
4. simless
Enter your choice (1/2/3/4):
```

Depending on the option that you select you will be further asked for details on the simulation you want to run, such as what model or applications you want to run and for what period.

> **NOTE:** you are asked to edit the document in vim, start the edition by typing 'i', edit it and save it by typing ':wq'.
> If you have any doubts when editing the main configuration file, read carefully the keys description in `docs/source/keys_main.rst`.

You will be popped up with a pdf of the structure o fthe workflow that you just have chosen. In this moment the experiment is as well visible [in the GUI](https://climatedt-wf.csc.fi/).

3. After your experiment is defined and you are fine with what you see in the image or in the GUI, you can do `autosubmit run $expid` to run your experiment.

> **Congratulations: you have created your minimal workflow! To know more about advanced workflow features, keep reading the README or dig into the [general documentation](https://earth.bsc.es/gitlab/digital-twins/de_340-2/workflow/-/wikis/home#docs).**

### Run your experiment

If you want to update the git repository refresh your experiment (equivalent to a git pull):

> **BE CAREFUL!** This command will overwrite any changes in the local project folder.
> Note that this is doing the same thing that the `autosubmit create` did in a previous
> step, but `autosubmit create` only refreshes the git repository the first time it is
> executed:

```bash
autosubmit refresh <expid>
```

Then you need autosubmit to create the updated the workflow files again:

```bash
autosubmit create <expid> -v -np
```

This resets the status of all the jobs, so if you do not want to run everything from
the beginning again, you can set the status of tasks, for example:

```bash
autosubmit setstatus a002 -fl "a002_LOCAL_SETUP a002_SYNCHRONIZE a002_REMOTE_SETUP" -t COMPLETED -s
```

`-fl` is for filter, so you filter them by job name now, `-t` is for target status(?)
so, we set them to `COMPLETED` here. `-s` is for save, which is needed to save the
results to disk.

You can add a `-np` for “no plot” to most of the commands to not have the error with
missing `xdg-open`, etc.

## Documentation

Basic instructions to execute the workflow with Autosubmit can be found in [How to run](https://earth.bsc.es/gitlab/digital-twins/de_340-2/workflow/-/wikis/How-to-run). The rest of the documentation is available in the same [Wiki](https://earth.bsc.es/gitlab/digital-twins/de_340-2/workflow/-/wikis/home).

To build the online version of the documentation you must clone the repo (`git clone https://earth.bsc.es/gitlab/digital-twins/de_340-2/workflow.git`) and:

```bash
# Maybe activate a venv or conda environment?
make docs-html
```

This will build the documentation in the folder `docs/build/html`. To access the documentation you can then click on `index.html` which will open the webpage docs (or open `index.html` in your favourite browser, e.g. `firefox index.html`), or start a web server and server it locally, for instance:

```bash
cd $workflow_code_root/
python -m http.server -d docs/build/html/ 8000
```

And then navigate to <http://localhost:8000/>.

The documentation can also be built in pdf, using:

```bash
make docs-latexpdf
```

This will build the documentation in `docs/build/latex/climatedtworkflow.pdf`.

## Contributing

This workflow is work in progress, and suggestions and contributions are greatly appreciated. If you have a suggestion, desire some new feature, or detect a bug, please:

1. Open an issue in this GitLab explaining it.
2. Once the issue is completely defined and assigned to someone, it will be tagged with the `to do` label.
3. Once someone is working in the issue, it will be tagged with the `working on` label.

If you want to develop yourself:

1. Create an experiment in the VM.
2. Set up your environment like explained in [Development environment](https://earth.bsc.es/gitlab/digital-twins/de_340-2/workflow/-/blob/main/docs/source/getting_started.rst).
3. Create a branch locally and make the changes that you want to implement. Make use of the [Shell Style Guide](urlhttps://google.github.io/styleguide/shellguide.html).
4. Test your changes.
5. Once you tested the workflow, add, commit and push your changes into a new branch. You can manually run all the pre-commit actions (for linting the yaml files, for example) by running `pre-commit run --all-files`.
6. Create a merge request. The pipeline will apply [Shellcheck](https://www.shellcheck.net/), `shfmt` and [ruff](https://docs.astral.sh/ruff/) to your code.
7. The workflow developers will test it and merge.

A Makefile with various tests, coverage analytics, formatting, etc. is available. As the tests use a ClimateDT container, GitHub access tokens are required. Ask for access to the [ClimateDT Container Recipes](https://github.com/DestinE-Climate-DT/ContainerRecipes) repository and follow the instructions to set up the token auth environment correctly.

If you modified template Shell scripts, please remember to run the tests:

```bash
make test
```

The command above binds the current directory to the `/code` directory inside the container. It is recommended to run the command above (or `bats` directly) from the project root directory (e.g. from `./workflow/`).

If you want to see the code coverage for the current tests, you can use:

```bash
make coverage
```

This command uses a `Docker` container with both `bats`, support libraries (`bats-assert` and `bats-support`), and `kcov` installed. It will create a local folder `./coverage/` with the HTML coverage report produced
by `kcov`. You can visualize it by opening `./coverage/index.html` in a browser.

## Contact us

For any doubts or issues you can contact the workflow developers:

- Miguel Castrillo (Activity leader): **miguel.castrillo[@]bsc.es**
- Leo Arriola (ICON Workflow developer): **leo.arriola[@]bsc.es**
- Sebastian Beyer (IFS-FESOM Workflow developer): **sebastian.beyer[@]awi.de**
- Francesc Roura Adserias (Applications Workflow developer): **francesc.roura[@]bsc.es**
- Ivan Alsina (Applications workflow developer): **ivan.alsina[@]bsc.es**
- Aina Gaya (IFS-NEMO Workflow developer): **aina.gayayavila[@]bsc.es**
- Damien McClain (IFS-NEMO workflow developer): **damien.mcclain[@]bsc.es**

Main Autosubmit support:

- Daniel Beltrán **daniel.beltran[@]bsc.es**
- Bruno de Paula Kinoshita **bruno.depaulakinoshita[@]bsc.es**

[Link to the Autosubmit tutorial & Hands-on](https://wiki.eduuni.fi/display/cscRDIcollaboration/Autosubmit+introductory+session)

## License

Copyright 2022-2025 European Union (represented by the European Commission)

The ClimateDT workflow is distributed as open-source software under Apache 2.0 License. The copyright owner is the European Union, represented by the European Commission. The development of the ClimateDT workflow has been funded by the European Union through Contract DE_340_CSC - Destination Earth Programme Climate Adaptation Digital Twin (Climate DT). Further info can be found at <https://destine.ecmwf.int/> and <https://destination-earth.eu/>
