import pandas
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt


def show_type_of_data(df):
    cat_data = ["color", "director_name", "actor_2_name", "genres", "actor_1_name", "movie_title", "actor_3_name", "plot_keywords", "movie_imdb_link", "language", "country", "content_rating"]

    for c in df.columns:
        if c in cat_data:
            print(c)
            print("Categorical")
            print(" Num of unique values: " + str(df[c].nunique()))
            print("---------")
        else:
            print(c)
            print("Continuous")
            print(" range: " + str(df[c].min()) + " - " + str(df[c].max()))
            print("----------------------")


def print_counts_of_values(df):
    cat_data = ["color", "director_name", "actor_2_name", "genres", "actor_1_name", "movie_title", "actor_3_name", "plot_keywords", "movie_imdb_link", "language", "country", "content_rating"]
    for c in df.columns:
        if c in cat_data:
            # make a list of counts and then values
            counts_values = zip(pandas.value_counts(df[c]).index.tolist(),
                                pandas.value_counts(df[c]).tolist())
            counts_values.sort(key=lambda x: x[1])
            print(counts_values)
            print("-------------------")


def clean_data(df):
    # Strip weird characters from movie titles
    for i in df.index:
        df.at[i, "movie_title"] = df.at[i, "movie_title"][:-2]

    # Drop all rows with any missing values
    for column in df.columns:
        df = df[pandas.notnull(df[column])]

    # Recalculate the indexs
    df = df.reset_index(drop=True)

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def group_countries(df):
    for i in df.index:
        if df.at[i, "country"] != "USA":
            df.at[i, "country"] = "Non USA"

    return df


def graph_genre_means(df):
    genres = ["Action", "Adventure", "Animation", "Biography", "Comedy", "Crime", "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "Mystery", "Romance", "Sci-Fi", "Sport", "Thriller", "War", "Western"]
    means = []

    for g in genres:
        print(g)
        scores = []
        for i in df.index:
            if g in df.at[i, "genres"]:
                scores.append(df.at[i, "imdb_score"])
        if len(scores) != 0:
            means.append(sum(scores)/len(scores))

    print(genres)
    print(means)

    objects = genres
    y_pos = np.arange(len(objects))
    performance = means

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.ylabel('IMDB Score')
    plt.title('Mean IMDB Score')

    plt.show()


def remove_columns(df):
    # Drop imdb link, movie title and plot_keywords
    df = df.drop("movie_imdb_link", 1)
    df = df.drop("plot_keywords", 1)
    df = df.drop("movie_title", 1)

    # drop names
    df = df.drop("director_name", 1)
    df = df.drop("actor_1_name", 1)
    df = df.drop("actor_2_name", 1)
    df = df.drop("actor_3_name", 1)

    # drop genres
    df = df.drop("genres", 1)

    # drop color and language
    df = df.drop("color", 1)
    df = df.drop("language", 1)

    # Recalculate the indexs
    df = df.reset_index(drop=True)

    return df


def main():
    # Read in dataset
    df = pandas.read_csv('movie_metadata.csv', index_col=None)

    df = clean_data(df)

    df = remove_columns(df)

    df = group_countries(df)

    print_counts_of_values(df)

    show_type_of_data(df)

    # write the new cleaned data set
    df.to_csv("cleaned_dataset.csv")


if __name__ == '__main__':
    main()
