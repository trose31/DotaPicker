//Initialise global variables and add event listeners to pre-existing buttons

const radPickBut = document.getElementById("radpick");
const dirPickBut = document.getElementById("dirpick");
const swapRecBut = document.getElementById("recommend");
const resetBut = document.getElementById("wipe");

radPickBut.addEventListener("click", function(){team = "rad"});
dirPickBut.addEventListener("click", function(){team = "dir"});
swapRecBut.addEventListener("click", swap);
resetBut.addEventListener("click", wipe);

const pickBoxes = document.getElementsByClassName("boxpick");
const sess = new onnx.InferenceSession();
const loadingModelPromise = sess.loadModel("onnx_model.onnx");

var evals = new Array(125);
var evalBoxes = new Array(125);
var evalScoreDisp = new Array(125);
var recommend = 0;

var team = "rad";
var radTeam = ["neutral","neutral","neutral","neutral","neutral"];
var dirTeam = ["neutral","neutral","neutral","neutral","neutral"];

/*Create, texture and add event listeners to the box showing each individual hero
which was quicker to do in the script than in the HTML/CSS file. */

const d4 = document.getElementById("d4");
for (i = 0; i < 125; i++)
{
	evalBoxes[i] = document.createElement("d4"+String(i));
	styleD4(evalBoxes[i]);
	d4.appendChild(evalBoxes[i]);
	
	evalScoreDisp[i] = document.createElement("d4"+String(i)+"eval");
	evalBoxes[i].appendChild(evalScoreDisp[i])
	styleD4Child(evalScoreDisp[i])

	evals[i] = [0, i];
	setup(pickBoxes[i]);
}

function styleD4(obj)  //Set the CSS for a recommendation box
{
	obj.style.position = "absolute";
	obj.style.backgroundSize = "cover";
	obj.style.height = "90px";
	obj.style.width = "175px";
	obj.style.left = "2px";
}

function styleD4Child(obj) //Set the CSS for the text of a recommendation block, which is set up as a child object
{
	obj.style.position = "absolute";
	obj.style.height = "90px";
	obj.style.width = "85px";
	obj.style.fontSize = "25px";
	obj.style.color = "white";
}

function swap()  //Swap which team the network is recommending picks for
{
	recommend = (recommend + 1) % 2
	swapRecBut.textContent = ["Recommend Dire", "Recommend Radiant"][recommend]

	neuralPredict()
}

function wipe() //Reset all picks
{
	radTeam = ["neutral","neutral","neutral","neutral","neutral"];
	dirTeam = ["neutral","neutral","neutral","neutral","neutral"];
	update()

	for (j=0; j<125;j++)
	{
		evalBoxes[j].style.height = "0px";
		evalBoxes[j].style.width = "0px";
		evalScoreDisp[j].innerHTML = "";
	}
}

function setup(obj)  //Add event listeners to the hero display boxes
{
	obj.addEventListener("mouseover",function(){enlarge(obj)})
	obj.addEventListener("mouseout", function(){resize(obj)})
	obj.addEventListener("click", function(){neuralPredict(obj)})
	texture(obj, obj.id)
	obj.zIndex = 1;
}


function pick(obj)  //Select a hero for a team
{
	for (i = 0; i < 5; i++)
	{
		if (radTeam[i]=="neutral" && team == "rad")
		{
			radTeam[i] = obj.id.substring(2,obj.id.length);
			break;
		}		
		else if (dirTeam[i]=="neutral" && team == "dir")
		{
			dirTeam[i] = obj.id.substring(2,obj.id.length);
			break;
		}
	}
}

function update()  //Update the display of already picked heroes
{
	for (i = 0; i < 5; i++)
	{
		texture(document.getElementById("c"+String(i+1)), radTeam[i]);
		texture(document.getElementById("e"+String(i+1)), dirTeam[i]);
	}

}

function texture(obj, id)
{
	obj.style.backgroundImage = "url('images/id"+id+".jpg')";
}


async function neuralPredict(obj)
{
	if (obj!=null)  //If neuralPredict is called after swapping the team of the recommendation, it is not given a hero to log as picked, so obj = null
	{
		pick(obj)
		update()
	}

	var rawInput = new Float32Array(250);

	for (i = 0; i < 5; i++)  //Creating and combining the one-hot encodings of the already picked heroes
	{
		rawInput[parseInt(radTeam[i])]=1
		rawInput[parseInt(dirTeam[i])+125]=1
	}

	for (id = 0; id < 125; id++)	//Simulate each possible pick by adding it to the rawInput, and outputing the evaluation by the network
	{
		var trueID = evals[id][1]
		var preActivation = rawInput[trueID+125*(recommend)]
		rawInput[trueID+125*(recommend)] = 1

		var input = new onnx.Tensor(rawInput, "float32", [1,250]);
		var outputMap = await sess.run([input]);
		var prediction = outputMap.values().next().value.data;

		evals[id][0] = Math.round((prediction[recommend]-0.5)*1000)/10;  //The network output is a probility distribution: [Radiant win, Dire win], extract the relevant probability
		rawInput[trueID+125*(recommend)] = preActivation
	}

	evaluationDisp();
}

function evaluationDisp()		//Update the displayed recommendations to the newly calculated values
{
	evals.sort(([a, b], [c, d]) => c - a || d - b);

	for (j=0; j<125;j++)
	{
		id = evals[j][1]
		score = evals[j][0]

		evalBoxes[id].style.top = String(j*93+25)+"px";
		evalBoxes[id].style.height = "90px";
		evalBoxes[id].style.width = "175px";
		
		evalScoreDisp[id].innerHTML = String(score);
		evalScoreDisp[id].style.color = "rgb("+String(2*j)+","+String(250-2*j)+",50)" 	//The evaluation is shown as greener for the better picks

		texture(evalBoxes[id], String(id));
	}
}

function enlarge(obj) //Enlarges a hero portrait box when a cursor is hovering over it
{
	obj.style.height = "150px";
	obj.style.width = "100px";
	obj.style.backgroundSize = "cover";
	obj.style.zIndex = 100;
}

function resize(obj)
{
	obj.style.height = "60px";
	obj.style.width = "36px";
	obj.style.backgroundSize = "60px 90px";
	obj.style.zIndex = 1;
}
