import HTTP, FileIO
using Dash
using DashHtmlComponents
using DashCoreComponents

# 1. Why would that be useful?
# In many cases, it is expected to publish the output of a piece of work to the web.
# It may also be needed to make a project available to a larger audience or just to interactively
# tinker with a workflow to get it working just right.

# A WebApp is a great tool to set up to not only achieve this but also make it look prettier
# with a vast array of tools desgined for the web already availbable.
# Typically this requires the knowledge of a lot of different frameworks that can make it all work,
# but that is typically if one needs to know how to work with these technologies (Write JavaScript? Never).
# Dash.jl makes it so that there is no additional language/ know-how of the web required
# to develop complex and interactive webapps in a simple manner.

# It may also be used to producionize or produtize a trained model to be used by a larger
# number of people without needing to how it works.

# 2. How would I make a local server
# Dash.jl is easy to use to understand and easy to demonstrate
# A scaled up system would use a more traditional webserver behind a
# proxy such as nginx. This can be done using Mux.jl. This is a demostration
# of the concept and workflow.

# 3. How does Dash.jl work
# This is a Julia port of Dash which is a very popular framework developed by the
# plotly team. It is used to create web apps without needing any Javascript experience.
# It makes use of callbacks to handle queries between the server and the user.
# All of the HTML primitives are available to us to make the template and interactively
# manage the frontend depending on what the user input is.

# 4. What will we do?
# We will train a neural network with Flux, and have a user defined file loaded and inferred
# through Dash.jl

# Let's Get Started
# We will start with defining our neural network. For the purposes of the talk, we
# will keep this fairly easy and have a simple CNN that we will use to learn the MNIST dataset.
# The network itself can be stored centrally and can be distributed to where you will serve the app from.

using Images
using Flux, Flux.Data.MNIST, Flux.NNlib
using BSON
using BSON: @load

# For the demo, we will train our model, but in production we only need to worry about
# the saved model and its pretrained weights, not the actual logic of what the ML
# pipeline is doing. They are completely separated, to make sure it so that no specialist knowledge is needed to run the webapp.

# Only when you want to train the neural net.
# include("train.jl")

# We load the model, its pretrained weights, and make sure eveything is set up to be run.
@load "model_conv.bson" model

ps = BSON.load("mnist_conv2.bson")[:params]
Flux.loadparams!(model, ps)

# We now have a trained model!

# We will use the Foundation CSS library for this training. But it can be your favourite CSS stylesheet
stylesheets = ["https://cdn.jsdelivr.net/npm/foundation-sites@6.4.3/dist/css/foundation-float.min.css",
         "https://cdn.jsdelivr.net/npm/foundation-sites@6.4.3/dist/css/foundation-prototype.min.css",
         "https://cdn.jsdelivr.net/npm/foundation-sites@6.4.3/dist/css/foundation-rtl.min.css"]

# As you can see, we have a bunch of CSS files that we can simply reuse in our app.
# To define the app, we create a `dash` object. The API is very similar to how the python version of
# plotly behaves, with some helpful sugar mixed in to make the whole process smoother.

# Note that the argument is the title of the app. This is displayed in the title bar.
# It additionally accepts many other keyword arguments. Here we give Dashboards the CSS we had chosen earlier
# and the folder from which it will pick up assets from.

# The body of the do-block holds in the base skeleton of what our app is supposed to do.
# Notice how we create `h1` and `divs` similar to how you would expect them to be in HTML.

app = dash(external_stylesheets = stylesheets, assets_folder = "assets")

app.layout =  html_div(id = "main graph") do
  html_h1("Say Hello to Dash.jl", style = (textAlign = "center",)),
  html_h3("A Julia interface for Plotly Dash", style = (textAlign = "center",), className = "subheader"),
  html_div() do
    html_br()

    # We annotate our different objects with `id`s that we can use inside our callbacks
    # to control user defined values in here to make our app responsive.
    # We will also give our input field a default empty string
    html_div(className = "row") do
      html_div(className = "columns small-3 small-centered") do
        dcc_input(id = "imgpath", value="", type = "text")
      end
    end,
    html_div(id = "imgpathdiv")
  end
end


isurl(x::String) = startswith(x, "http")

# A `callback!` just bound a function to be executed from one object to the next when a certain
# change occurs. Here the call to imgpath is bound to imgpathdiv the empty div whose contents
# we want to manipulate with changes to the input div. We get the input value of the input object
# is the argument to our function.

callback!(app, Output("imgpathdiv", "children"), Input("imgpath", "value")) do input_value

  # First we handle the default empty string. Note that all inputs we will receive from Dashboards
  # will be strings in nature, and it is up to us to convert them into the data types we want.
  # Although, we can give it julia data types which it will try to convert appropriately.
  # If we find an empty string in the input box, we display a helpful message and a random image,
  # to make the app look nicer and more lively.
  input_value == "" && return html_picture(draggable = "true") do
    html_div() do
      html_h3("Enter a valid File Name or URL", className = "columns small-3 small-centered")
    end,

    html_div(className = "row") do
      html_div(className = "columns small-3 small-centered") do
        html_img(src = "https://picsum.photos/200/300")
      end
    end
  end

  isurl(input_value) || any(endswith.(input_value, [".jpg", ".jpeg", ".png"])) || return

  # We now load our input image (based on the user defined value), and resize it to fit how our
  # model expects inputs to be shaped.
  img_path = if isurl(input_value)
    HTTP.download(input_value, joinpath("assets", "temp.jpg"))
    joinpath("assets", "temp.jpg")
  else
    joinpath("assets", input_value)
  end
  @show img_path

  !isfile(img_path) && return html_h1("Invalid File Name")
  
  img = FileIO.load(img_path)
  img = Images.imresize(img, 28,28)
  img = reshape(channelview(img), size(img)..., 1, 1)

  # Note that since this is just a regular julia function at the same time, we can use all the
  # goodies that we get with writing a general julia script. Here, we are logging the output to
  # the REPL, but using LoggingExtras.jl and the Logging module in julia, one can write
  # descriptive logs files to monitor the activities happening inside our app
  @show model(img)

  # To elucidate our point, it might be helpful for the user to see a snapshot of the image that
  # they just sent through our ML model. To do that, we can simply use the html constructors
  # by passing to them where in our asset registry the different files exist.
  # We can also pass a named tuple to the style argument to control the style aspects as we do with
  # HTML.
  nt = (width = "100px", height = "100px", textAlign = "center")
  html_div(className = "row") do
    html_div(className = "columns small-3 small-centered") do
      html_picture(draggable = "true") do
        html_img(src = isurl(input_value) ? input_value : joinpath("assets", input_value), style = nt)
      end
    end
  end,

  # We are close now. We finally need to show users the result of our computation. We will do that
  # in a couple simple steps. First we will take the image that we have read in our memory and
  # do a forward pass in our trained model. Note that if we so wanted, we can annotate the `model`
  # and our input `img` with `Flux.gpu` if we have GPUs available on the machine to get better
  # performance. We don't need any extra configuration.

  # Through `model(img)`, we have the values of the probabilities of the image being a number between
  # 0 and 9. Now we want to display it as a bar graph. We use plotly to do our analytical plotting.
  # The dcc_graph function allows us to simply set the values on the x-axis being 0-9 and y-axis as
  # probabilities. We can further give it different keywords to make it plenty obvious what it is that
  # we are trying to implement.
  dcc_graph(
    id = "example-graph",
    figure = (
      data = [
        # (x = collect(0:9), y = [4, 1, 2], type = "bar", name = "SF"),
        (x = collect(0:9), y = model(img), type = "bar", name = "MNIST"),
      ],
      layout = (title = "Dash Data Visualization",)
    )
  )
end

run_server(app, "0.0.0.0", parse(Int,ARGS[1]))
