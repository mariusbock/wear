# WEAR Challenge 2024

THE WEAR dataset [1] is an outdoor sports dataset for inertial-based human activity recognition (HAR). The dataset comprises data from 18 participants performing a total of 18 different workout activities with untrimmed inertial (acceleration) data recorded at 10 different outside locations. WEAR provides a challenging prediction scenario marked by purposely introduced activity variations as well as an overall small information overlap across modalities.

In 2024 we introduce the WEAR Dataset challenge, an activity recognition challenge using the inertial data provided by the original dataset publication of the WEAR dataset [1]. Challenge participants of the challenge are faced to predict a yet unreleased test dataset of newly and re-recorded participants. Results will be presented at the HASCA (Human Activity Sensing Corpus and Applications) Workshop at UbiComp/ ISWC 2024.

## Prizes
**tbd**

## Deadlines
**tbd**

## Registration
Each team must send a registration email to **tbd**@anon.com as soon as possible but not later than **tbd**, stating the:

- Name of the team
- Full Names of the participants in the team
- Affiliations of each participant (non-affiliated persons are also encouraged to participate)
- E-mail adress of one participant representing the team

## HASCA Workshop

To be part of the final ranking, participants will be required to submit a detailed paper to the HASCA workshop. The paper should contain technical description of the processing pipeline, the algorithms and the results achieved during the development/ training phase. The submissions must follow the HASCA format (see **tbd**), but with a page limit between 3 and 6 pages. Publishing of the paper also requires that at least one team member is registered for the HASCA workshop (http://hasca2023.hasc.jp/).

## Submission of predictions on the test dataset

Participants must submit a plain text prediction file following the format "{team_name}.txt" with "team_name" being the chosen name of the team (as specified during registration). Specifically, the submitted file must contain a matrix of size **tbd** rows x 3 columns (first column being the subject id, second column being the timestamp, and third column being label prediction). An example submission is provided as part of the data download. Each team’s predictions must be submitted online by sending an email to **tbd**@anon.com, in which there should be a download link of the team's prediction file, using services such as Dropbox, Google Drive, etc. In case a team cannot provide a link using some file sharing service, they should contact the organizers via email **tbd**@anon.com. To be part of the final ranking, participants will be required to publish a detailed paper in the proceedings of the HASCA workshop. The date for the paper submission is **tbd**. All the papers must be formatted using **tbd**. The template is available at **tbd**. Submissions do not need to be anonymous. Submission is electronic, using precision submission system. The submission site is open at https://new.precisionconference.com/submissions (select **tbd** and push Go button). See the image below.

[add image here]

## Dataset format

The data download is divided into two parts: the original WEAR dataset as introduced in [1] and an unpublished, private test dataset. The original dataset comprises of outdoor workouts of 18 participants. Each workout is divided across multiple session. In total more than 15 hours were recorded at 10 outdoor locations. The private test dataset features unpublished, newly recorded outdoor workouts of **tbd** additional participants as well as rerecordings of **tbd** participants (``sbj_tbd``, ``sbj_tbd`` and ``sbj_tbd``) 

The training data contains the raw sensor data of the 18 partipants which were part of the original WEAR dataset [1]. The sensors were placed at four body locations (right wrist, left wrist, right ankle and left ankle). Each sensor sampled 3D-accelerometer data. During all recording sessions sensor orientation was fixed according to one pre-defined sensor placement. Each sampled data record is labeled as one of the 19 possible activities. 

[TABLE HERE]

The testing data is structured the same way as the training data. Sensor were placed on the same body locations and in the same orientation as in the training data. Participants are tasked to classify each data record as one of the 19 activities.

## Downloads

### Data 
- Original WEAR data [...GB] (download)
- Test data [...GB] (download)

### Submission example
This is an example of how a submission should look like: (download)

## Rules
Some of the main rules are listed below. The detailed rules are contained in the following document.

- Eligibility
    - You do not work in or collaborate with the WEAR dataset project (http://mariusbock.github.io/wear/)
    - If you submit an entry, but are not qualified to enter the contest, this entry is voluntary. The organizers reserve the right to evaluate it for scientific purposes. If you are not qualified to submit a contest entry and still choose to submit one, under no circumstances will such entries qualify for sponsored prizes.
- Entry
    - **Registration (see above):** as soon as possible but not later than **tbd**.
    - **Challenge:** Participants will submit prediction results on test data.
    - **Workshop paper:** To be part of the final ranking, participants will be required to publish a detailed paper in the proceedings of the HASCA workshop (http://hasca2023.hasc.jp/); The dates will be set during the competition. Publishing of the paper also requires that at least one team member is registered for the HASCA workshop (http://hasca2023.hasc.jp/).
    - **Submission:** The participants’ predictions should be submitted online by sending an email to **tbd***@gmail.com, in which there should be a link to the predictions file, using services such as Dropbox, Google Drive, etc. In case the participants cannot provide link using some file sharing service, they should contact the organizers via email **tbd**@gmail.com.
    - **Only one single submission is allowed per team.** The same person cannot be in multiple teams, except if that person is a supervisor. The number of supervisors is limited to 3 per team.

## Contact
All inquiries should be directed to: **tbd**@gmail.com

## Organizers
Marius Bock, University of Siegen (GER)
Dr. Kristof Van Laerhoven, University of Siegen (GER)
Dr. Michael Moeller, University of Siegen (GER)
Alexander Hoelzemann, University of Siegen (GER)
Christina Runkel, University of Cambridge (UK)
Dr. Mathias Ciliberto, University of Sussex (UK)

## References
[1] M. Bock, H. Kuehne, K. Van Laerhoven, M. Moeller, “WEAR: An Outdoor Sports Dataset for Wearable and Egocentric Activity Recognition,” CoRR abs/2304.05088, 2023. [link](https://arxiv.org/abs/2304.05088)
