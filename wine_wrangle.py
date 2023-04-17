{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "85341e11-c1e3-489f-920c-105313bd65c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "25415ce9-49fd-4881-8ed2-67137a53161d",
   "metadata": {},
   "outputs": [],
   "source": [
    "############ Acquire #####################\n",
    "def acquire_wine(df):\n",
    "    \"\"\" This function will access and concat the red wine and white wine csvs acquired from the data.world dataframes\n",
    "    for preparation\"\"\"\n",
    "    if os.path.isfile('wine.csv'):\n",
    "        return pd.read_csv('wine.csv')\n",
    "    else: \n",
    "        # Calling in my dfs from csv link\n",
    "        red = pd.read_csv('https://query.data.world/s/azffrkwaoqlfrd3srbnuwjp24hvlj4?dws=00000')\n",
    "        white = df = pd.read_csv('https://query.data.world/s/6ao5pdvepveo2qeeafwdfia6bl5mou?dws=00000')\n",
    "        \n",
    "        # Adding 'type' categories on both dataframes before concating\n",
    "        red['type'] = 'red'\n",
    "        white['type'] = 'white'\n",
    "        \n",
    "        # The two become one\n",
    "        df = wine = pd.concat([red, white], index=False)\n",
    "        return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "0a6ac0fb-9d81-4438-94d7-b673b8b781ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "############ Prepare #####################\n",
    "def prep_wine(df):\n",
    "    \"\"\" This function will concat the red wine and white wine csvs acquired from the data.world dataframes\n",
    "    and prepare them for exploration and analysis\"\"\"\n",
    "    # Creating variable to categorize quality\n",
    "    df['quality_bins'] = pd.cut(df.quality,[0,5,7,9], labels=['low_quality', 'mid_quality', 'high_quality'])\n",
    "    # Creating dummy variables for type\n",
    "    dummies = pd.get_dummies(data=df[['type']], dummy_na= False, drop_first=False)\n",
    "    df = pd.concat([df, dummies], axis = 1)\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "716d6813-30b3-4452-8159-3117ff8a9789",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "############ Prepare #####################\n",
    "def split_wine(df, target):\n",
    "    '''\n",
    "    take in a DataFrame return train, validate, test split on zillow DataFrame.\n",
    "    '''\n",
    "# Reminder: I don't need to stratify in regression. I don't remember why, but Madeleine said \n",
    "# it\n",
    "     train, test = train_test_split(df, test_size=.2, random_state=123, stratify=df[target])\n",
    "    train, val = train_test_split(train, test_size=.30, random_state=123, stratify=train[target])\n",
    "    return train, val, test"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
