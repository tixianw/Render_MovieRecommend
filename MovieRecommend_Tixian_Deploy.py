import pandas as pd
import numpy as np
import random

from logging import debug
import dash
import dash_bootstrap_components as dbc
import requests
import io
from dash import Input, Output, dcc, html
from dash.dependencies import ALL, State

def read_from_raw(url):
    # urlData = requests.get(url).content
    # df = pd.read_csv(io.StringIO(urlData.decode('utf-8')))
    # df.set_index('Unnamed: 0', inplace=True)
    # df.index.name = None
    response = requests.get(url, timeout=10)
    df = pd.read_csv(io.StringIO(response.text), index_col=0)
    return df


url_movies = 'https://github.com/tixianw/Render_MovieRecommend/raw/main/Data/movies.csv'
url_Recommender_Genre = 'https://github.com/tixianw/Render_MovieRecommend/raw/main/Data/Recommender_Genre.csv'
url_Similarity_mtx_top30 = 'https://github.com/tixianw/Render_MovieRecommend/raw/main/Data/Similarity_mtx_top30.csv'
movies = pd.read_csv(url_movies, index_col=0) # read_from_raw(url_movies) # 
Recommender_Genre = pd.read_csv(url_Recommender_Genre, index_col=0) # read_from_raw(url_Recommender_Genre) # 
Similarity_mtx_top30 = pd.read_csv(url_Similarity_mtx_top30, index_col=0) # read_from_raw(url_Similarity_mtx_top30) # 
S_top30 = Similarity_mtx_top30.values
col_names = Similarity_mtx_top30.columns.tolist()
n_movies = len(S_top30)

genres = list(
    sorted(set([genre for genres in movies.Genres.unique() for genre in genres.split("|")]))
)

def get_displayed_movies(n=100):
    return movies.sample(n)
    # return movies.head(100)

### System I
def get_popular_movies(genre: str, n=10):
    recommend_genre = Recommender_Genre[Recommender_Genre['InputGenre']==genre]
    movie_id_genres_double = recommend_genre['MovieID'][:2*n].to_list()
    half = n//2
    idx_sample = sorted(random.sample(range(half,2*n), half))
    movie_id_genres = movie_id_genres_double[:n//2] + list(np.array(movie_id_genres_double)[idx_sample])
    indices = []
    for i in range(n):
        indices.append(movies[movies['MovieID']==movie_id_genres[i]].index[0])
    return movies.loc[indices]
    # # if genre == genres[1]:
    # #     return movies.head(10)
    # # else: 
    # #     return movies[10:20]

### System II
def transform_rating(new_user_rating):
    new_rating = np.full(n_movies, np.nan)
    idx_rated = []
    for i, key in enumerate(new_user_rating.keys()):
        idx_rated.append(col_names.index('m'+str(key)))
        new_rating[idx_rated[-1]] = new_user_rating[key]
    return new_rating, idx_rated

def make_pred(one_rating):
    pred = np.zeros(n_movies)
    new_rating, idx_rated = transform_rating(one_rating)
    for i in range(n_movies):
        if i in idx_rated:
            pred[i] = np.nan
            continue
        idx_neighbor = np.where(~np.isnan(S_top30[i]))[0] # Idx_Sort[i] # 
        idx_valid = np.intersect1d(idx_rated, idx_neighbor)
        if len(idx_valid):
            pred[i] = S_top30[i, idx_valid] @ new_rating[idx_valid] / np.sum(S_top30[i, idx_valid])
    return pred

def get_recommended_movies(new_user_ratings, n=10):
    print('check rated movies:', new_user_ratings)
    pred = make_pred(new_user_ratings)
    idx_nonnan = np.where(~np.isnan(pred))[0]
    idx_sort = np.argsort(pred[idx_nonnan])[::-1][:n]
    id_pred = np.array(col_names)[idx_nonnan[idx_sort]]
    indices = []
    for i in range(n):
        indices.append(movies[movies['MovieID']==int(id_pred[i][1:])].index[0])
    return movies.loc[indices]
    # return movies.head(10)


'''
    Create the App using Dash
'''

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP], 
               suppress_callback_exceptions=True)
server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem", 
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H3("Movie Recommender", className="display-8"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("System 1 - Recommend by Genre", href="/", active="exact"),
                dbc.NavLink("System 2 - Recommend by Rating", href="/system-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

def get_movie_card(movie, with_rating=False):
    return html.Div(
        dbc.Card(
            [
                dbc.CardImg(
                    src=f"https://liangfgithub.github.io/MovieImages/{movie.MovieID}.jpg?raw=true",
                    top=True,
                ),
                dbc.CardBody(
                    [
                        html.H6(movie.Title, className="card-title text-center"),
                    ]
                ),
            ]
            + (
                [
                    dcc.RadioItems(
                        options=[{"label": x, "value": x} for x in [str(i) for i in range(1,6)]],
                        inline=True, ##
                        className="text-center",
                        id={"type": "movie_rating", "movie_id": movie.MovieID},
                        inputClassName="m-1",
                        labelClassName="px-1",
                    )
                ]
                if with_rating
                else []
            ),
            className="h-100",
        ),
        className="col mb-4",
    )

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page_content(pathname):
    if pathname == "/":
        return html.Div(
            [
                html.H1("Select a genre"),
                dcc.Dropdown(
                    id="genre-dropdown",
                    options=[{"label": k, "value": k} for k in genres],
                    value=None,
                    className="mb-4",
                ),
                html.Div(id="genre-output", className=""),
            ]
        )
    elif pathname == "/system-2":
        movies = get_displayed_movies()
        return html.Div(
            [
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    html.H1("Rate some movies below"),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        children=[
                                            "Get recommendations ",
                                            html.I(className="bi bi-emoji-heart-eyes-fill"),
                                        ],
                                        size="lg",
                                        className="btn-success",
                                        id="button-recommend",
                                    ),
                                    className="p-0",
                                ),
                            ],
                            className="sticky-top bg-white py-2",
                        ),
                        html.Div(
                            [
                                get_movie_card(movie, with_rating=True)
                                for idx, movie in movies.iterrows()
                            ],
                            className="row row-cols-1 row-cols-5",
                            id="rating-movies",
                            style={
                                'height': '600px',  # Set the height of the box
                                'overflowY': 'auto',  # Enable vertical scrolling
                                'border': '1px solid #ddd',  # Add a border for better visibility
                                'padding': '10px',  # Add some padding for better appearance
                            }
                        ),
                    ],
                    id="rate-movie-container",
                ),
                html.H1(
                    "Your recommendations", id="your-recommendation",  style={"display": "none"}
                ),
                dcc.Loading(
                    [
                        dcc.Link(
                            "Try rating some more movies!", href="/system-2", refresh=True, className="mb-2 d-block"
                        ),
                        html.Div(
                            className="row row-cols-1 row-cols-5",
                            id="recommended-movies",
                        ),
                    ],
                    type="circle",
                ),
            ]
        )

@app.callback(Output("genre-output", "children"), Input("genre-dropdown", "value"))
def update_output(genre):
    if genre is None:
        return html.Div()
    else: 
        return [
            dbc.Row(
                [
                    html.Div(
                        [
                            *[
                                get_movie_card(movie)
                                for idx, movie in get_popular_movies(genre).iterrows()
                            ],
                        ],
                        className="row row-cols-1 row-cols-5",
                    ),
                ]
            ),
        ]

@app.callback(
    Output("rate-movie-container", "style"),
    Output("your-recommendation", "style"),
    [Input("button-recommend", "n_clicks")],
    prevent_initial_call=True,
)    
def on_recommend_button_clicked(n):
    return {"display": "none"}, {"display": "block"}

@app.callback(
    Output("recommended-movies", "children"),
    [Input("rate-movie-container", "style")],
    [
        State({"type": "movie_rating", "movie_id": ALL}, "value"),
        State({"type": "movie_rating", "movie_id": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def on_getting_recommendations(style, ratings, ids):
    rating_input = {
        ids[i]["movie_id"]: int(rating) for i, rating in enumerate(ratings) if rating is not None
    }
  
    recommended_movies = get_recommended_movies(rating_input)
 
    return [get_movie_card(movie) for idx, movie in recommended_movies.iterrows()]


@app.callback(
    Output("button-recommend", "disabled"),
    Input({"type": "movie_rating", "movie_id": ALL}, "value"),
)
def update_button_recommened_visibility(values):
    return not list(filter(None, values))

if __name__ == '__main__':
    app.run_server(debug=True) # port=8080, 