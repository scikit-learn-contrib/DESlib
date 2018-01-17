Contributing to DESlib
========================

You can contribute to the project in several ways:

- Reporting bugs
- Requesting features
- Improving the documentation
- Adding examples to use the library
- Implementing new features and fixing bugs

Reporting Bugs and requesting features:
---------------------------------------

We use Github issues to track all bugs and feature requests; feel free to
open an issue if you have found a bug or wish to see a new feature implemented.
Before opening a new issue, please check if the issue is not being currently addressed:
[Issues](https://github.com/Menelau/DESlib/issues)

For reporting bugs:

-  Include information of your working environment. This information
   can be found by running the following code snippet:

   ```python
   import platform; print(platform.platform())
   import sys; print("Python", sys.version)
   import numpy; print("NumPy", numpy.__version__)
   import scipy; print("SciPy", scipy.__version__)
   import sklearn; print("Scikit-Learn", sklearn.__version__)
   ```

-  Include a [reproducible](https://stackoverflow.com/help/mcve) code snippet
   or link to a [gist](https://gist.github.com). If an exception is raised,
   please provide the traceback.

Documentation:
--------------

We are glad to accept any sort of documentation: function docstrings,
reStructuredText documents (like this one), tutorials, etc.
reStructuredText documents live in the source code repository under the
doc/ directory.

You can edit the documentation using any text editor and then generate
the HTML output by typing ``make html`` from the doc/ directory.
Alternatively, ``make`` can be used to quickly generate the
documentation without the example gallery. The resulting HTML files will
be placed in _build/html/ and are viewable in a web browser. See the
README file in the doc/ directory for more information.

For building the documentation, you will need to install sphinx and sphinx_rtd_theme. This
can be easily done by installing the requirements for development using the following command:

pip install -r requirements-dev.txt

Contributing with code:
-----------------------

The preferred way to contribute is to fork the main repository to your account:

1. Fork the [project repository](https://github.com/Menelau/DESlib):
   click on the 'Fork' button near the top of the page. This creates
   a copy of the code under your account on the GitHub server.

2. Clone this copy to your local disk:

        $ git clone git@github.com:YourLogin/DESlib.git
        $ cd DESlib

3. Install all requirements for development:

        $ pip install -r requirements-dev.txt
        $ pip install --editable .

4. Create a branch to hold your changes:

        $ git checkout -b branch_name

Where ``branch_name`` is the new feature or bug to be fixed. Do not work directly on the ``master`` branch.

5. Work on this copy on your computer using Git to do the version
   control. To record your changes in Git, then push them to GitHub with:

        $ git push -u origin branch_name

It is important to assert your code is well covered by test routines (coverage of at least 90%), well documented and
follows PEP8 guidelines.

6. Create a 'Pull request' to send your changes for review.

   If your pull request addresses an issue, please use the title to describe
   the issue and mention the issue number in the pull request description to
   ensure a link is created to the original issue.




