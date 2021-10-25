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
Assuming you're in the `go` directory you can:
```shell
$ go mod tidy # install the dependencies
$ go run *.go # run the program
```

### Python
Assuming you're in the `python` directory you can:
```shell
$ pip3 install -r requirements.txt  # install the dependencies (on a M1-Mac you need to install tensorflow-macos!)
$ python3 *.py                      # run the program
```