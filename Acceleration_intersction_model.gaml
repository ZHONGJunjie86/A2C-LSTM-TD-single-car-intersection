/**
* Name: simpleintersction
* Author: ZHONG Junjie
*/

model simpleintersction

global {   
	file shape_file_roads  <- file("../includes/intersection/easy/simple_network.shp") ;
	file shape_file_nodes  <- file("../includes/intersection/easy/simple_nodes.shp");//simple_nodes2
	geometry shape <- envelope(shape_file_roads);

	int nb_people <-0; //10
	int nb_bus <- 1;
	int time_to_set_offset <- 1;
	int episode <- 0;
	int total_episode<-10000;
	int done;  //終わり-1
	int time_target;

	node_agt starting_point; //agent species
	
	graph road_network;
	graph kagayaki_network;
	graph pana_east_network;
	graph kasayama_network;
	
	map kagayaki_route;
	map kasayama_route;
	map general_speed_map;
	
	path kagayaki_path;
	path kasayama_path;
	path pana_east_path;
	
	point t1;//スタート地点
	point t2;//ゴール地点
	
	file bus_shape_kagayaki  <- file('../includes/icons/vehicles/bus_blue.png');
	file bus_shape_kasayama  <- file('../includes/icons/vehicles/bus_green.png');
	file car_shape_empty  <- file('../includes/icons/vehicles/normal_red.png');
	
	init {  
		
		create road from: shape_file_roads with:[id::int(read("id")),nblanes::int(read("lanes")),maxspeed::int(read("maxspeed")),highway::string(read("highway")),
			kasayama::int(read("kasayama")),kagayaki::int(read("kagayaki")),pana_east::int(read("pana-east"))] {
			
		    //lanes <- 1;
		    maxspeed <-  60 + (rnd(20)-10)°m/°s;// (lanes = 1 ? 30.0 : (lanes = 2 ? 50.0 : 70.0)) °km/°h;
		    
		    if(kagayaki!=1){
		    	kagayaki <- 500;//重みを極端にする
		    }
		    if(kasayama!=1){
		    	kasayama <- 500;
		    }
		    if(pana_east!=1){
		    	pana_east <- 500;
		    }
		    switch oneway {
		    	match "no" {
		    		create road {
					  	lanes <- max([1, int (myself.lanes / 2.0)]);
						shape <- polyline(reverse(myself.shape.points));
						maxspeed <- myself.maxspeed;
						geom_display  <- myself.geom_display;
						linked_road <- myself;
						
						self.kagayaki <- myself.kagayaki; 
						myself.linked_road <- self;
						
						
						if(myself.kagayaki!=1){
					    	self.kagayaki <- 500;//重みを極端にする
					    }
					    if(myself.kasayama!=1){
					    	self.kasayama <- 500;
					    }
					    if(myself.pana_east!=1){
					    	self.pana_east <- 500;
					    }								
						
					  }
					  //lanes <- int(lanes /2.0 + 0.5);
				 }
		    	match "yes" {
		    		create road {
					  	lanes <- max([1, int (myself.lanes / 2.0)]);
						shape <- polyline(reverse(myself.shape.points));
						maxspeed <- myself.maxspeed;
						geom_display  <- myself.geom_display;
						linked_road <- myself;
						myself.linked_road <- self;
						self.kagayaki <- myself.kagayaki; 
						
						if(myself.kagayaki!=1){
					    	self.kagayaki <- 500;//重みを極端にする
					    }
					    if(myself.kasayama!=1){
					    	self.kasayama <- 500;
					    }
					    if(myself.pana_east!=1){
					    	self.pana_east <- 500;
					    }
					  }
					  //lanes <- int(lanes /2.0 + 0.5);
				 }
				 match "-1" {
				 	lanes <- 1;
				 	self.linked_road <- self;
				 	shape <- polyline(reverse(shape.points));
				}
			}
			geom_display <- shape+ (2.5 * lanes);
		    maxspeed <- 60 + (rnd(20)-10)°m/°s;//maxspeed <- (lanes = 1 ? 30.0 : (lanes = 2 ? 50.0 : 70.0)) °km/°h;
		}
		
		create node_agt from: shape_file_nodes with:[is_traffic_signal::(string(read("type")) = "traffic_signals"),type::(string(read("type")))];
		starting_point <- one_of(node_agt where each.is_traffic_signal);
		
		general_speed_map <- road as_map (each::(each.shape.perimeter / (each.maxspeed)));
		
		
		/*以下数行を書き換える */
		kagayaki_route <- road as_map(each::(each.kagayaki));//重みを
		kasayama_route <- road as_map(each::(each.kasayama));
		road_network <-  (as_driving_graph(road, node_agt))with_weights general_speed_map;
		
		t1 <- (node_agt(5)).location;   //スタート地点  //5 list 4 ?  
		t2 <- (node_agt(12)).location;  //ゴール地点  //12
		
		
				
		create people number: nb_people { 
			speed <- 30 #km /#h ;
			vehicle_length <- 3.0 #m;
			right_side_driving <- true;
			proba_lane_change_up <- 0.1 + (rnd(500) / 500);
			proba_lane_change_down <- 0.5+ (rnd(500) / 500);
			location <- one_of(node_agt).location;
			security_distance_coeff <- 4.0;//(1.5 - rnd(1000) / 1000);  
			proba_respect_priorities <- 1.0 - rnd(200/1000);
			proba_respect_stops <- [0.1];
			proba_block_node <- 0.0;
			proba_use_linked_road <- 0.0;
			max_acceleration <- 0.5 + rnd(500) / 1000;
			speed_coeff <- 1.2 - (rnd(400) / 1000);
		}
		
		
		create bus number: nb_bus { 
			time_target<- 12+ rnd(13);
			max_speed <- 330°m/°s;
			real_speed <- 0 °m/°s;
			target_speed<-9 + rnd(3)°m/°s; //30
			vehicle_length <- 3.0 #m;
			location <- t1;
			time_pass <- 0;
			done <- 0;
			over <- 0;
			max_acceleration <- 0.0 ;
		}	
			
	}
    reflex stop_simulation when: episode = total_episode {
        do pause ;
    } 
	
} 

//路線バスの基本形エージェント
  species bus skills: [advanced_driving,RSkill] { 
	  rgb color <- rnd_color(255);
	  bool green_light_checked <-false;
  	  node_agt target; 
	  node_agt true_target;
	  node_agt bus_start;
	  int n <-1;
	  int m <- rnd(n);//乱数でバスのルートを決定
      // 自作
      int time_pass;
	  int first_time <- 1;
	  int over;
      int check_receive <- 0;
      int random_node;
	  int a_flag_checked_pass_light <- nil;
	  unknown read_python;
	  unknown pause;
	  unknown clear;
      float reward <- 0.0;
      float acceleration ; 
      float target_speed;   
	  float elapsed_time_ratio;
	  float distance_all;
	  float distance_passed <-0.0;
	  float distance_left <-0.0;
	  point pre_point;
	  file Rcode_pause<-text_file("D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/R_pause.txt"); //file Rcode_clear<-text_file("D:/Software/GamaWorkspace/Python/R_clear.txt");
      file Rcode_read<-text_file("D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/R_read.txt");  //D:\Software\PythonWork\GAMA_python\A2C-TD-single-car-intersection\GAMA_read.py
	
	action road_weight{
		if(m=0){
			 road_network <- road_network with_weights kagayaki_route;//重みを地図に適応		
			 current_path <- compute_path(graph: road_network,target: target);
		 }
		 if(m=1){
		   road_network <- road_network with_weights kasayama_route;//重みを地図に適応
		   current_path <- compute_path(graph: road_network,target: target);
		 } 
		 int length_nodes <- length(targets);
		 int sum_nodes <- 0;
		 loop while: (sum_nodes+1 < length_nodes) {
		     distance_all <- distance_all + (targets at sum_nodes)distance_to(targets at (sum_nodes+1));
		     sum_nodes <- sum_nodes+1;
		 }
		 distance_left <- distance_all;
		 distance_left<-max (distance_left,distance_to_goal);
		 if(round(distance_left/100)>= 0){
		     time_target <- int((distance_left/100)*5)+ rnd(3);  //key
		 }
		 else{
		 	time_target <- rnd(2) + 4;
		 }
		 
	}
	action distance_cal{
		distance_passed<- distance_passed + (location distance_to pre_point);
		pre_point <- location;
		distance_left <- distance_all - distance_passed;
		//if(distance_left<0){distance_left <- 0;}
		distance_left<-max (distance_left,distance_to_goal);
	}
	action reward_calculate{
    	if(time_pass != 0){
		  if(real_speed<=target_speed and real_speed != 0){
			  reward <- 0.05 - 0.033*(target_speed/real_speed) + (time_pass > time_target ? -0.013 : 0.005); 
		  }
		  else if(real_speed = 0){
		  	reward <- -0.5;
		  }
		  else{ //超速
			  reward <-  0.05 - 0.04*(real_speed/target_speed) + (time_pass > time_target ? -0.01 : 0.005); //36 sigmiod 
		  }
	    }
	    else{reward<-0.0;}
	}
    action Python_A3C {
    	 do reward_calculate;
    	 do distance_cal;
         elapsed_time_ratio <- time_pass/time_target;
         over <- (episode < total_episode ? 0 : 1);
         //write "episode:"+episode+"reward: "+reward+" . over:"+over+". done:"+ done;
         //write "rs "+real_speed+"ts " + target_speed+"etr"+ elapsed_time_ratio+"dtg"+ distance_to_goal+"r"+reward+"d"+done+"o"+over;
    	 save [real_speed/10, target_speed/10, elapsed_time_ratio, distance_left/100,reward,done,over] //time_pass
    	                                to: "D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/GAMA_intersection_data_1.csv"  type: "csv" header: false; 
    	 save [real_speed/10, target_speed/10, elapsed_time_ratio, distance_left/100,reward,done,over] 
    	                                to: "D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/GAMA_intersection_data_2.csv"  type: "csv" header: false; 
    	                                
         //write "read";
	     loop s over:Rcode_read.contents{read_python<- R_eval(s);}
	     check_receive <- read_python at 0;
	     //waiting for python
	     loop while: check_receive = 0 {
	         loop s over:Rcode_pause.contents{pause<- R_eval(s);}
	         loop s over:Rcode_read.contents{read_python<- R_eval(s);}  
	         check_receive <- read_python at 0;
	      }
	      //wait over. caculate
	      acceleration  <- read_python at 1;     write"check_receive acceleration:"+acceleration;
          //clear write "clear";loop s over:Rcode_clear.contents{clear<- R_eval(s);}  //seems useless
	      save [0] to: "D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/python_AC_1.csv"  type: "csv" header: false;//
	      save [0] to: "D:/Software/PythonWork/GAMA_python/A2C-TD-single-car-intersection/GAMA_R/python_AC_2.csv"  type: "csv" header: false;
      }    
    				        
	reflex change when :current_path = nil{		
		a_flag_checked_pass_light <- nil;
		write "a_flag_checked_pass_light: "+ a_flag_checked_pass_light;
		location <- any_location_in(node_agt(5));
		final_target<- any_location_in(node_agt(12));
	}
	//信号に引っかかった後の処理  等红灯。。。
	reflex time_to_go when: a_flag_checked_pass_light != 0 and green_light_checked = true {
		if(true_target != nil){
			target <- true_target;
			true_target <- nil;
		}
		//do road_weight;
	}	
	
	//目的地（終着バスターミナル）についた時の処理  && 最初の時   location distance_to final_target
	 reflex time_to_restart when:(distance_left)<=1 or first_time = 1 {  //a_flag_checked_by_light = 0 and checked = false
	     if(first_time != 1){
             done <- 1;
             write "done!!";
          	 do Python_A3C;
		     episode <- episode + 1 ; //加到1000，算完最后结果
         }
         else{
         	first_time <- 0;
            do startR;
         }
         write "計算";
         real_speed <- 0;
         time_pass <- 0;
        
         random_node <- int(rnd(12)); //rnd(
         loop while: random_node = 5{random_node <- int(rnd(12));}
         target<- node_agt(random_node);
         true_target <- node_agt(random_node);
		 final_target <- node_agt(random_node).location;	
		 a_flag_checked_pass_light <- 1;
		 location <- any_location_in(node_agt(5)); 
		 pre_point<-any_location_in(node_agt(5)); 
		 distance_passed <- 0.0;
		 distance_all<- 0.0;
		 reward<-0.0;
		 do road_weight;
         target_speed<- distance_left/time_target;//9 + rnd(3)°m/°s;
         write "target_time "+ time_target+"target_speed "+target_speed+" distance_left "+distance_left;
	} 
	
	reflex move when: current_path != nil or a_flag_checked_pass_light != 0 {//道が決まり、目的地が決まれば動く  
	    if (episode<total_episode and done = 0 ){
           do Python_A3C;
           real_speed <- real_speed + acceleration ;  //1;//
           if(real_speed<=0){real_speed <- 0;}
        }
        if(episode<total_episode and done = 1){
           done<-0;
           do Python_A3C;
           real_speed <- real_speed + acceleration; 
           if(real_speed<=0){real_speed <- 0;}
        }
		do drive;
	    time_pass <- time_pass + 1;
	}
	aspect car3D {
		if (current_road) != nil {
			point loc <- calcul_loc();
			draw box(vehicle_length, 1,1) at: loc rotate:  heading color: color;
			draw triangle(0.5) depth: 1.5 at: loc rotate:  heading + 90 color: color;	
		}
	}
	
	aspect icon {
		point loc <- calcul_loc();
			if(m =0){
			draw bus_shape_kagayaki size: vehicle_length   at: loc rotate: heading + 90 ;
			}
			if(m = 1)
				{
			draw bus_shape_kasayama size: vehicle_length   at: loc rotate: heading + 90 ;	
					}
		}
	
	point calcul_loc {
		float val <- (road(current_road).lanes - current_lane) + 0.5;
		val <- on_linked_road ? val * - 1 : val;
		if (val = 0) {
			return location; 
		} else {
			return (location + {cos(heading + 90) * val, sin(heading + 90) * val});
		}
	}

}

species road skills: [skill_road] { 
    int id;
    int nblanes;
    string highway;
	string oneway;
	geometry geom_display;
	road riverse;
	int kasayama;
	int kagayaki;
	int pana_east;
	bool observation_mode <- true; //交通量観察モード　挙動が重いときはこれをfalseに
	int flow <- 0; //交通量
	list temp1 <- self.all_agents; // t = n -1 の交通量保持のためのリスト
	
	//交通量計測のためのメソッド	
	reflex when :observation_mode  {
		
		if((length(all_agents) - length(temp1)) > 0){
			flow <- flow + length(all_agents) - length(temp1); 		
		}	
		temp1 <- self.all_agents;
	}
	
	aspect geom {    
		draw geom_display border:  #gray  color: #gray ;
	}  
}

species node_agt skills: [skill_road_node] {
	bool is_traffic_signal;
	string type;
	int cycle <- 100; //サイクル長
	float split <- 0.5 ; //スプリット
	int counter;
	int offset <- 0;
	bool is_blue <- true;
	list<road> current_accessible ; //現通行可
	list<road> current_forbidden ; //現通行不可
	list<node_agt> adjoin_node;//隣接交差点[東,西,南,北]
	string mode <- "independence";
	agent c1; //信号制御で止める車1
	agent c2; //信号制御で止める車2
	agent f1;
	agent f2;

	//オフセット設定（広域信号制御の際に使用）
	reflex set_offset when:time = time_to_set_offset and is_traffic_signal{
		starting_point.mode <- "start";
		loop i from: 0 to: length(starting_point.adjoin_node)-1 {
			starting_point.adjoin_node[i].offset <- 0;
		}	
	}
	
	//起点モード（広域信号制御の際に使用）
	reflex set_adjoinnode when: time = 0{
		if(length(self.roads_out) >1){
			loop i from: 0 to: length(self.roads_out) - 1 {
				self.adjoin_node <- self.adjoin_node + [node_agt(road(roads_out[i]).target_node)] where each.is_traffic_signal;
			}
		}
	}
	
	
	//現示の切り替えタイミング
	reflex start when:counter = 10 {//counter >= cycle*split+offset  even()
			counter <- 0; 
			is_blue <- !is_blue; //
	} 
	
	//4叉路用信号制御処理
	reflex stop4 when:is_traffic_signal and length(roads_in) = 4
	{
		counter <- counter + 1;
		
		if(contains(bus,c1)){ //c1是bus吗
			bus(c1).green_light_checked <- false;
		}
	
		if(contains(bus,c2)){
			bus(c2).green_light_checked <- false;
		}
		
		if(contains(people,c1)){
			people(c1).green_light_checked <- false;
		} 
		if(contains(people,c2)){
			people(c2).green_light_checked <- false;
		}
		
		c1 <- nil;
		c2 <- nil;
		
		//現示処理（通行権付与処理）  绿灯能通行的路
		if (is_blue) {		
		    current_accessible <- [road(roads_in[0]),road(roads_in[2])];
		    current_forbidden <- [road(roads_in[1]),road(roads_in[3])];
		}
		else{       		
			current_accessible <- [road(roads_in[1]),road(roads_in[3])];
			current_forbidden <- [road(roads_in[0]),road(roads_in[2])];
		}
		 
		if(length(current_accessible) != 0 or length(current_forbidden) != 0 ){
			if(length(current_accessible[0].all_agents) != 0 ){  //current_shows[0]----road(roads_in[0]) or road(roads_in[1])
				c1 <- current_accessible[0].all_agents[0]; //c1 可以走的路的第一条 上面的所有agent
				if(contains(bus,c1)){
					bus(c1).true_target <- bus(c1).target;
					//bus(c1).final_target <- any_location_in(self);
					bus(c1).a_flag_checked_pass_light <- int(any_location_in(self));
					bus(c1).green_light_checked <- true;
				}
				if(contains(people,c1)){
					people(c1).true_target <- people(c1).target;
					people(c1).final_target <- any_location_in(self);
					people(c1).green_light_checked <- true;
				}
				//write "c1: "+c1+"a_flag_checked_by_light: " + bus(c1).a_flag_checked_pass_light+" current_path"+bus(c1).current_path;
			 }
			                                               //current_shows[1]----road(roads_in[2]) or road(roads_in[3])
		    if(length(current_accessible[1].all_agents) != 0){  //all_agents: the list of agents on the road 
				c2 <- current_accessible[1].all_agents[0]; //c2 可以走的路的第二条 上面的所有agent
				write "c2: "+c2;
				if(contains(bus,c2)){
					bus(c2).true_target <- bus(c2).target;
					//bus(c2).final_target <- any_location_in(self);
					bus(c2).a_flag_checked_pass_light <- int(any_location_in(self));
					bus(c2).green_light_checked <- true; 
				}
				
				if(contains(people,c2)){
					people(c2).true_target <- people(c2).target;
					people(c2).final_target <- any_location_in(self);
					people(c2).green_light_checked <- true;
				}
			 }
			 //不能通行
			 if(length(current_forbidden[0].all_agents) != 0){  
				f1 <- current_forbidden[0].all_agents[0]; 
				if(contains(bus,f1)){
					bus(f1).true_target <- bus(f1).target;
					//bus(c2).final_target <- any_location_in(self);
					bus(f1).a_flag_checked_pass_light <- 0;//int(any_location_in(self));
					bus(f1).green_light_checked <- false; 
				}
				
			     if(contains(people,f1)){
					people(f1).true_target <- people(f1).target;
					//people(c2).final_target <- any_location_in(self);
					people(f1).green_light_checked <- false;
				}
			   }
			   if(length(current_forbidden[1].all_agents) != 0){ 
				    f2 <- current_forbidden[1].all_agents[0]; 
				    write "f2: "+f2;
				    if(contains(bus,c2)){
					    bus(f2).true_target <- bus(f2).target;
					    //bus(c2).final_target <- any_location_in(self);
					    bus(f2).a_flag_checked_pass_light <- 0;//int(any_location_in(self));
					    bus(f2).green_light_checked <- false; 
				}
				
				if(contains(people,f2)){
					people(f2).true_target <- people(f2).target;
					people(f2).final_target <- 0;//any_location_in(self);
					people(f2).green_light_checked <- false;
				}
			 }
		}
	}
		
	//3叉路用信号制御処理		
	reflex stop3 when:is_traffic_signal and  length(roads_in) =  3{
		
		counter <- counter + 1;

		//以下初期化
		if(contains(bus,c1)){
			bus(c1).green_light_checked <- false;
		}
	
		if(contains(bus,c2)){
			bus(c2).green_light_checked <- false;
		}
		
		if(contains(people,c1)){
			people(c1).green_light_checked <- false;
		} 
		if(contains(people,c2)){
			people(c2).green_light_checked <- false;
		}
	

		c1 <- nil;
		c2 <- nil;
		
		//現示処理（通行権付与処理）
		if (is_blue) {		
				current_accessible <- [road(roads_in[0])];				
				}else{
				current_accessible <- [road(roads_in[1]),road(roads_in[2])]; 
			}
			
		//現示の道路に車がいない時の処理
		if(length(current_accessible) != 0){
			
			if(length(current_accessible[0].all_agents) != 0 ){
				c1 <- current_accessible[0].all_agents[0]; 
				
				if(contains(bus,c1)){
				    bus(c1).true_target <- bus(c1).target;
					//bus(c1).final_target <- any_location_in(self);
					bus(c1).a_flag_checked_pass_light <- int(any_location_in(self));
					bus(c1).green_light_checked <- true;
				}
				
				if(contains(people,c1)){
					people(c1).true_target <- people(c1).target;
					people(c1).final_target <- any_location_in(self);
					people(c1).green_light_checked <- true;
				}
			}
			
			//現示の道路が二本以上の時
			if(length(current_accessible) > 1){
				if(length(current_accessible[1].all_agents) != 0){
					c2 <- current_accessible[1].all_agents[0];
					
					if(contains(bus,c2)){
						bus(c2).true_target <- bus(c2).target;
						//bus(c2).final_target <-  any_location_in(self);
						bus(c2).a_flag_checked_pass_light <- int(any_location_in(self));
						bus(c2).green_light_checked <- true;
					}
					if(contains(people,c2)){
						people(c2).true_target <- people(c2).target;
						people(c2).final_target <- any_location_in(self);
						people(c2).green_light_checked <- true;
					}	
				}
			}
		}
	}
	
	aspect geom3D {
		if (is_traffic_signal) {	
			draw box(1,1,10) color:rgb("black");
			draw sphere(5) at: {location.x,location.y,12} color: is_blue ? #green : #red;
		}
	}
}

	
species people skills: [advanced_driving] { 
	rgb color <- rnd_color(255);
	bool green_light_checked;
	node_agt target;
	node_agt true_target;
	
	reflex change when :current_path = nil{
		final_target <- nil;
	}
	
	
	reflex time_to_go when: final_target = nil and green_light_checked = false{
		
		target <- one_of(node_agt);
		if(true_target != nil){
		target <- true_target;
		true_target <- nil;
		}
		road_network <- road_network with_weights general_speed_map;//重みを地図に適応	
		current_path <- compute_path(graph: road_network, target: target );
		if (current_path = nil ) {
			final_target <- nil;
		}
	}
	
	reflex time_to_go_true when: final_target = nil and green_light_checked = true{	
		current_path <- compute_path(graph: road_network, target: target );
	}
	
	reflex move when: current_path != nil and final_target != nil {
		real_speed <- real_speed + 4 ;
		do drive;
	}
	aspect car3D {
		if (current_road) != nil {
			point loc <- calcul_loc();
			draw box(vehicle_length, 1,1) at: loc rotate:  heading color: color;
			draw triangle(0.5) depth: 1.5 at: loc rotate:  heading + 90 color: color;	
		}
	} 
	
	aspect icon {
		point loc <- calcul_loc();
			draw car_shape_empty size: vehicle_length   at: loc rotate: heading + 90 ;
	}
	
	point calcul_loc {
		float val <- (road(current_road).lanes - current_lane) + 0.5;
		val <- on_linked_road ? val * - 1 : val;
		if (val = 0) {
			return location; 
		} else {
			return (location + {cos(heading + 90) * val, sin(heading + 90) * val});
		}
	}
} 


experiment traffic_simulation type: gui {
	float minimum_cycle_duration<-0.04#s;
	output {
		display city_display type: opengl{
			species road aspect: geom refresh: false;
			species node_agt aspect: geom3D;
			species people aspect: icon;
			species bus aspect: icon;
		}
	}
}