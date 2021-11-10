using Clustering
using CSV
using DataFrames
using Plots


function elbow_method(data)
    #=
    Apply the elbow method to get optimal number of clusters.
    Plots the elbow graph and asks the user to input the visual optimal.
    =#
    wcss = []
    for i in 1:10
        R = kmeans(data, i)
        push!(wcss, R.totalcost)
    end

    plt = plot(1:10, wcss, legend=false)
    plt = scatter!(1:10, wcss, legend=false)
    display(plt)
    print("Optimal clusters in 'Elbow Method'-plot: ")
    return parse(Int64, readline())
end


dataset = CSV.read("../data/clustering/clustering.csv", DataFrame; header=0)  # Load Dataset
data = collect(Matrix(dataset)')
n_clusters = elbow_method(data)
#n_clusters = 4
result = kmeans(data, n_clusters)

# Plot Results
plt = scatter(dataset.Column1, dataset.Column2, marker_z=result.assignments, color=:lightrainbow, legend=false)
display(plt)
print("Press enter to close plot ")
readline()