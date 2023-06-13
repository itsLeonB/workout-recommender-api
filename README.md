# Workout Recommender API

This API is based on the [ML API Template](https://github.com/kaenova/ml-api-template) by [@kaenova](https://github.com/kaenova). Thanks bang!

## Running the server

1. Clone the repository:
```sh
git clone https://github.com/your-username/your-repository.git
```

2. Install the required libraries:
```sh
pip install -r requirements.txt
```

3. Run the server:
```sh
python main.py
```

## Retrieving recommendations

Access the recommendations endpoint:
```sh
http://localhost:8080/recommendations/{workout_index}
```
Give the selected workout index on {workout_index}. The endpoint will return a list of indices of the top 3 workouts recommended.

### For testing and additional documentation, access the /docs endpoint