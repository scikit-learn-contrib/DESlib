.. _packaging:

Releasing a new version
=======================

Publishing new version involves:

1) Updating the version numbers and creating a new tag in git (which also updates the "stable" version of the documentation)
2) Creating the distribution (.tar.gz and wheel files), and uploading them to pypi

Some important things to have in mind:
 * Read the "Packaging and Distributing Projects" guide: https://packaging.python.org/tutorials/distributing-packages/
 * The version numbers (in setup.py and __init__.py) are used as metadata for pypi and for the readthedocs documentation - pay attention to them or some things can break. In general, you should be working on a version such as "0.2.dev". You then rename it to "0.2" and create a tag "v0.2". After you finish everything, you update the version to "0.3.dev" to indicate that new developments are being made for the next version.


Step-by-step process
--------------------


* Create an account in PyPi production: https://pypi.org/ and test: https://test.pypi.org/
* Make sure you have twine installed:

 .. code-block:: bash

  pip install twine

* Update version on setup.py (e.g. "0.1")
* Update version on deslib/__init__.py
* Create tag: :code:`git tag <version>` (example: "git tag 'v0.1'")
* Push the tag :code:`git push origin <version>`
* Create the source and wheels distributions

 .. code-block:: bash

    python setup.py sdist # source distribution
    python setup.py bdist_wheel # wheel distribution for current python version

* Upload to test pypi and check

  - uploading the package:

  .. code-block:: bash

    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

  - Note: if you do this multiple times (e.g. to fix an issue), you will need to rename the files under the "dist" folder: a filename can only be submitted once to pypi. You may also need to manually delete the "source" version of the distribution, since there can only be one source file per version of the software

  - Test an installation from the testing pypi environment.

  .. code-block:: bash

     conda create -y -n testdes python=3
     source activate testdes
     pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple deslib
     conda remove -y --name testdes --all  #remove temporary environment

* Upload to production pypi

  .. code-block:: bash

     twine upload dist/*

* Mark the new stable version to be built on readthedocs:

 - Go to https://readthedocs.org/projects/deslib/versions/, find the new tag and click "Edit". Mark the "active" checkbox and save.

* Update version on setup.py and __init.py__ to mention the new version in development (e.g. "0.2.dev")


Note #1: Read the docs is automatically updated:

* When a new commit is done in master (this updates the "master" version)
* When a new tag is pushed to github (this updates the "stable" version)  -> This seems to not aways work - it is better to check

Note #2: The documentation automatically links to source files for the methods/classes. This only works if the tag is pushed to github, and matches the __version__ variable in __init.py__. Example:
__version__ = "0.1" and the tag being:
git tag "v0.1"
