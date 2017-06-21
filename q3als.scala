import breeze.linalg._
import org.apache.spark.HashPartitioner
//spark-shell -i als.scala to run this code
//SPARK_SUBMIT_OPTS="-XX:MaxPermSize=4g" spark-shell -i als.scala


//Implementation of sec 14.3 Distributed Alternating least squares from stanford Distributed Algorithms and Optimization tutorial. 

//loads ratings from file
val ratings = sc.textFile("hdfs://cshadoop1.utdallas.edu//user/bxm142230/input/ratings.dat").map(l => (l.split("::")(0),l.split("::")(1),l.split("::")(2))) 

// counts unique movies
val itemCount = ratings.map(x=>x._2).distinct.count 

// counts unique user
val userCount = ratings.map(x=>x._1).distinct.count 

// get distinct movies
val items = ratings.map(x=>x._2).distinct   

// get distinct user
val users = ratings.map(x=>x._1).distinct  

// latent factor
val k= 5  

//create item latent vectors
val itemMatrix = items.map(x=> (x,DenseVector.zeros[Double](k)))   
//Initialize the values to 0.5
// generated a latent vector for each item using movie id as key Array((movie_id,densevector)) e.g (2,DenseVector(0.5, 0.5, 0.5, 0.5, 0.5)
var myitemMatrix = itemMatrix.map(x => (x._1,x._2(0 to k-1):=0.5)).partitionBy(new HashPartitioner(10)).persist  

//create user latent vectors
val userMatrix = users.map(x=> (x,DenseVector.zeros[Double](k)))
//Initialize the values to 0.5
// generate latent vector for each user using user id as key Array((userid,densevector)) e.g (2,DenseVector(0.5, 0.5, 0.5, 0.5, 0.5)
var myuserMatrix = userMatrix.map(x => (x._1,x._2(0 to k-1):=0.5)).partitionBy(new HashPartitioner(10)).persist 

// group rating by items. Elements of type org.apache.spark.rdd.RDD[(String, (String, String))] (itemid,(userid,rating)) e.g  (1,(2,3))
val ratingByItem = sc.broadcast(ratings.map(x => (x._2,(x._1,x._3)))) 

// group rating by user.  Elements of type org.apache.spark.rdd.RDD[(String, (String, String))] (userid,(item,rating)) e.g  (1,(3,5)) 
val ratingByUser = sc.broadcast(ratings.map(x => (x._1,(x._2,x._3))))




var i =0
for( i <- 1 to 10){
	// regularization factor which is lambda.
	val regfactor = 1.0 
val regMatrix = DenseMatrix.zeros[Double](k,k)  //generate an diagonal matrix with dimension k by k
//filling in the diagonal values for the reqularization matrix.
regMatrix(0,::) := DenseVector(regfactor,0,0,0,0).t 
regMatrix(1,::) := DenseVector(0,regfactor,0,0,0).t 
regMatrix(2,::) := DenseVector(0,0,regfactor,0,0).t 
regMatrix(3,::) := DenseVector(0,0,0,regfactor,0).t 
regMatrix(4,::) := DenseVector(0,0,0,0,regfactor).t 
        
        var movieJoinedData = myitemMatrix.join(ratingByItem.value)
        var tempUserMatrixRDD = movieJoinedData.map(q => 
                                 {
                                    var movie = q._1
                                    var (user,rating) = q._2._2
                                     var tempDenseVector = q._2._1
                   		     var tempTransposeVector = tempDenseVector.t
                                     var tempDenseMatrix = tempDenseVector * tempTransposeVector
                                     (user,tempDenseMatrix)
                                 }
                   ).reduceByKey(_+_).map(q => (q._1,inv(q._2+regMatrix))) 

       var tempUserVectorRDD = movieJoinedData.map(q => 
                                 {
                                    var movie = q._1
                                    var (user,rating) = q._2._2
                                    var tempDenseVector = q._2._1
                                     
                                    var secondTempVector = tempDenseVector :* rating.toDouble
                                    (user,secondTempVector)
                                 }
                   ).reduceByKey(_+_)

       var finalUserVectorRDD =   tempUserMatrixRDD.join(tempUserVectorRDD).map(p => (p._1,p._2._1*p._2._2))
       myuserMatrix = finalUserVectorRDD.partitionBy(new HashPartitioner(10)).persist 

        var userJoinedData = myuserMatrix.join(ratingByUser.value)
        var tempMovieMatrixRDD = userJoinedData.map(q => 
                                 {
                                    var tempUser = q._1
                                    var (movie,rating) = q._2._2
                                     var tempDenseVector = q._2._1
                   		     var tempTransposeVector = tempDenseVector.t
                                     var tempDenseMatrix = tempDenseVector * tempTransposeVector
                                     (movie,tempDenseMatrix)
                                 }
                   ).reduceByKey(_+_).map(q => (q._1,inv(q._2+regMatrix))) 

       var tempMovieVectorRDD = userJoinedData.map(q => 
                                 {
                                     var tempUser = q._1
                                     var (movie,rating) = q._2._2
                                     var tempDenseVector = q._2._1
                                     
                                     var secondTempVector = tempDenseVector :* rating.toDouble
                                     (movie,secondTempVector)
                                 }
                   ).reduceByKey(_+_)

       var finalMovieVectorRDD =   tempMovieMatrixRDD.join(tempMovieVectorRDD).map(p => (p._1,p._2._1*p._2._2))
       myitemMatrix = finalMovieVectorRDD.partitionBy(new HashPartitioner(10)).persist 

}

/*
user 1 and movieid 914,
user 1757 and movieid 1777,
user 1759 and movieid 231.
*/

var pairsList = List(("1","914"),("1757","1777"),("1759","231"))
for((user,movie) <- pairsList)
{
   var userDenseVector = myuserMatrix.lookup(user)(0)
   var movieDenseVector = myitemMatrix.lookup(movie)(0)
   var predictedRating = userDenseVector.t * movieDenseVector
   println("==========================================================")
   println("Latent vector for user " + user + " : " + userDenseVector)
   println("Latent vector for movie " + movie + " : " + movieDenseVector)
   println("Predicted Rating by user " + user + " for movie " + movie + " : " + predictedRating)
}



