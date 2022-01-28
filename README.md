# ML languages comparison

In this repository you can find examples of ML solutions implemented in multiple languages.
## Description
For the Meetup we focus on 4 languages:
- Python
- Julia
- Go
- Rust

## Usage
Put training and test data in the data folder. To add a solution add a directory with the name of your language and put inside the code and instructions on how to run it.
***
## Running the code

### Go
Assuming you're in the `go` directory you can `cd` to the relevant directory for the solution you want and run it.
```shell
$ go mod tidy                       # install the dependencies
$ go run *.go                       # run the program
```
#### Kmeans
This implementation uses the `muesli` package, and was partly based on the cluster analysis shown by spatial-go [1].

#### Random forest
This implementation is based on the book "Machine Learning with Go" [2].

#### CNN
This implementation is based on the tutorial for a simple convolution neural net [3].

### Python
Assuming you're in the `python` directory you can:
```shell
$ pip3 install -r requirements.txt  # install the dependencies (on a M1-Mac you need to install tensorflow-macos!)
$ python3 <file_name>.py            # run the program
```

### Julia
Assuming you're in the `julia` directory you can:
```shell
$ julia requirements.jl             # install the dependencies
$ julia <file_name>.jl              # run the program
```

###  Rust
Install Rust, check https://www.rust-lang.org/tools/install

In the RUST ecosystem Cargo is your friend, it a package manager which handles all your dependencies and provides 
the BUILD tool for your RUST projects. But is handles basically all your development needs from building to testing all
your CI stuff! 

Starter, Hello World in RUST without typing code:

    $ cargo new HelloWorld
    $ cd HelloWorld
    $ cargo run

#### Kmeans
Based on Rusty-machine Crate

    $ cd $rust/kmeans/rusty-machine
    $ cargo run --release

#### Random Forest
Based on RustLearn Crate

    $ cd $rustl/randomforest/rustlearn
    $ cargo run --release

#### MNIST DNN/CNN

## References
1. Spatial-go.com. 2022. [online] Available at: <http://spatial-go.com/docs/cluster_analysis.html> [Last accessed 28 January 2022].
1. Whitenack, D. and Selvaraj, J., 2019. Machine Learning With Go. 2nd ed. Packt Publishing.
1. Gorgonia. 2022. Simple Convolution Neural Net (MNIST). [online] Available at: <https://gorgonia.org/tutorials/mnist/> [Last accessed 28 January 2022].