# Contribution Guide

**Please, follow these steps**

## Step 1: Forking and Installing quickvision

​1. Fork the repo to your own github account. Click the Fork button to
create your own repo copy under your GitHub account. Once forked, you're
responsible for keeping your repo copy up-to-date with the upstream
quickvision repo.

​2. Download a copy of your remote `<username>/quickvision` repo to your
local machine. This is the working directory where you will make
changes:

```bash
$ git clone https://github.com/<username>/quickvision
```

3.  Install the requirements. You may use miniconda or conda as well.

```bash
$ pip install -r requirements-test.txt
```

4. Install this package in develop mode. Go to root of this package and run the following:

```bash
$ python setup.py develop
```

## Step 2: Set up upstream repo

1.  Set the upstream to sync with this repo. This will keep you in sync
    with `quickvision` easily.

```bash
$ git remote add upstream https://github.com/Quick-AI/quickvision
```

2.  Updating your local repo: Pull the upstream (original) repo.

```bash
$ git checkout master
$ git pull upstream master
```

## Step 3: Creating a new branch

```bash
$ git checkout -b <feature-name>
$ git branch
 master 
 * <feature-name>: 
```

## Step 4: Make changes, and commit your file changes

Stage and commit your changes.

```
git add .
git commit -m "Your meaningful commit message for the change."
```

Add more commits, if necessary.

## Step 5: Submitting a Pull Request

#### 1. Create a pull request git

Upload your local branch to your remote GitHub repo
(`github.com/<username>/quickvision`)

```bash
git push origin <feature-name>
```

After the push completes, a message may display a URL to automatically
submit a pull request to the upstream repo. If not, go to the
quickvision main repo and GitHub will prompt you to create a pull
request.

#### 2. Confirm PR was created

Ensure your PR is listed
[here](https://github.com/Quick-AI/quickvision/pulls)

3.  Updating a PR:

    Same as before, normally push changes to your branch and the PR will get automatically updated.

    ```bash
    git commit -m "Updated the feature"
    git push origin <enter-branch-name-same-as-before>
    ```

* * * * *

## Reviewing Your PR

Maintainers and other contributors will review your pull request.
Please participate in the discussion and make the requested changes.
When your pull request is approved, it will be merged into the upstream quickvision repo.
