# Deploying Flux Models with Heroku

Steps to start:

* Clone this repo
* `$ cd utrain`
* `$ heroku login`
* Go to heroku.com - create a new app and give it a name
* Add a buildpack `https://github.com/DhairyaLGandhi/heroku-buildpack-julia.git`
* `$ git add .`
* `$ git commit -am "deploy with utrain2"`
* `$ git push heroku master`
* Wait for it to spin up the app
* Go to the URL
