<!-- -------------------------------------------- -->
 ### TO fix fatal issue due to different user in the system

'''
 $ git remote add origin https://github.com/rajarotomaker/pipeline.git
fatal: detected dubious ownership in repository at 'D:/raja/pipeline'
'D:/raja/pipeline' is owned by:
        COMP-AI/comp1 (S-1-5-21-3932468533-1321018147-1854945590-1002)
but the current user is:
        COMP-AI/Comp (S-1-5-21-3932468533-1321018147-1854945590-1003)
To add an exception for this directory, call:

        git config --global --add safe.directory D:/raja/pipeline
'''

# Run 
git config --global --add safe.directory D:/raja/pipeline

This adds the folder to Git’s safe list, and then Git stops blocking anything.

<!-- -------------------------------------------- -->
