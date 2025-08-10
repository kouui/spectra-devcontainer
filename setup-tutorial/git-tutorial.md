# Git Tutorial for contributors in SPECTRA project


**make sure you are in the project directory when you want to execute git commands**

## experiment with SPECTRA

for just experiment in jupyter notebook, make sure you are in the `master` branch to recieve the latest updates from the project.

if there are any updates, you can pull the latest changes from the remote repository by running the following command

```bash
git pull origin master
```

## development workflow


for development, you should create a new branch for your changes to avoid conflicts with the `master` branch.

first make sure you are in the `master` branch and pull the latest changes from the remote repository


```bash
git pull origin master
```

if you want to make changes to the code, you should first create a new branch for your changes to avoid conflicts with the ``master`` branch. (replace `my-feature-branch` with your actual branch name)

- if you are new to git, it is recommended to use the vscode GUI to manage branches and commits.

```bash
git checkout -b my-feature-branch
```

when you have made changes to the code, you can check the status of your local repository by running the following command

```bash
git status
```

add the files you want to commit to the staging area (replace `file1.py file2.py` with your actual file names)

```bash
git add file1.py file2.py
```

after adding the files to the staging area, you can commit your changes with a message (replace `my commit message` with your actual commit message)

```bash
git commit -m "my commit message"
```

if you want to push your changes to the remote repository, you can run the following command (replace `my-feature-branch` with your actual branch name)

```bash
git push origin my-feature-branch
```

after pushing your changes, send email to notify the project maintainers about your changes. 

once your changes are reviewed and approved, you can update your `master` branch for further development/experiment.

```bash
git checkout master
git pull origin master
```