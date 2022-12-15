import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


PATH='./results/'
RAND_RANGE=1000


class ActorNetwork(nn.Module):
    # actornetwork pass the test
    def __init__(self,state_dim,action_dim,n_conv=128,n_fc=128,n_fc1=128):
        super(ActorNetwork,self).__init__()
        self.s_dim=state_dim
        self.a_dim=action_dim
        self.vectorOutDim=n_conv
        self.scalarOutDim=n_fc
        self.numFcInput=2 * self.vectorOutDim * (self.s_dim[1]-4+1) + 3 * self.scalarOutDim + self.vectorOutDim*(self.a_dim-4+1)
        self.numFcOutput=n_fc1

        #-------------------define layer-------------------
        self.tConv1d=nn.Conv1d(1,self.vectorOutDim,4)

        self.dConv1d=nn.Conv1d(1,self.vectorOutDim,4)

        self.cConv1d=nn.Conv1d(1,self.vectorOutDim,4)

        self.bufferFc=nn.Linear(1,self.scalarOutDim)

        self.leftChunkFc=nn.Linear(1,self.scalarOutDim)

        self.bitrateFc=nn.Linear(1,self.scalarOutDim)

        self.fullyConnected=nn.Linear(self.numFcInput,self.numFcOutput)

        self.outputLayer=nn.Linear(self.numFcOutput,self.a_dim)
        #------------------init layer weight--------------------
        # tensorflow-1.12 uses glorot_uniform(also called xavier_uniform) to initialize weight
        # uses zero to initialize bias
        # Conv1d also use same initialize method 
        nn.init.xavier_uniform_(self.bufferFc.weight.data)
        nn.init.constant_(self.bufferFc.bias.data,0.0)
        nn.init.xavier_uniform_(self.leftChunkFc.weight.data)
        nn.init.constant_(self.leftChunkFc.bias.data,0.0)
        nn.init.xavier_uniform_(self.bitrateFc.weight.data)
        nn.init.constant_(self.bitrateFc.bias.data,0.0)
        nn.init.xavier_uniform_(self.fullyConnected.weight.data)
        nn.init.constant_(self.fullyConnected.bias.data,0.0)
        nn.init.xavier_uniform_(self.tConv1d.weight.data)
        nn.init.constant_(self.tConv1d.bias.data,0.0)
        nn.init.xavier_uniform_(self.dConv1d.weight.data)
        nn.init.constant_(self.dConv1d.bias.data,0.0)
        nn.init.xavier_normal_(self.cConv1d.weight.data)
        nn.init.constant_(self.cConv1d.bias.data,0.0)


    def forward(self,inputs):

        bitrateFcOut=F.relu(self.bitrateFc(inputs[:,0:1,-1]),inplace=True)

        bufferFcOut=F.relu(self.bufferFc(inputs[:,1:2,-1]),inplace=True)
 
        tConv1dOut=F.relu(self.tConv1d(inputs[:,2:3,:]),inplace=True)

        dConv1dOut=F.relu(self.dConv1d(inputs[:,3:4,:]),inplace=True)

        cConv1dOut=F.relu(self.cConv1d(inputs[:,4:5,:self.a_dim]),inplace=True)
       
        leftChunkFcOut=F.relu(self.leftChunkFc(inputs[:,5:6,-1]),inplace=True)

        t_flatten=tConv1dOut.view(tConv1dOut.shape[0],-1)

        d_flatten=dConv1dOut.view(dConv1dOut.shape[0],-1)

        c_flatten=cConv1dOut.view(dConv1dOut.shape[0],-1)

        fullyConnectedInput=torch.cat([bitrateFcOut,bufferFcOut,t_flatten,d_flatten,c_flatten,leftChunkFcOut],1)

        fcOutput=F.relu(self.fullyConnected(fullyConnectedInput),inplace=True)
        
        out=torch.softmax(self.outputLayer(fcOutput),dim=-1)

        return out


class CriticNetwork(nn.Module):
    # return a value V(s,a)
    # the dim of state is not considered
    def __init__(self,state_dim,a_dim,n_conv=128,n_fc=128,n_fc1=128):
        super(CriticNetwork,self).__init__()
        self.s_dim=state_dim
        self.a_dim=a_dim
        self.vectorOutDim=n_conv
        self.scalarOutDim=n_fc
        self.numFcInput=2 * self.vectorOutDim * (self.s_dim[1]-4+1) + 3 * self.scalarOutDim + self.vectorOutDim*(self.a_dim-4+1)
        self.numFcOutput=n_fc1

        #----------define layer----------------------
        self.tConv1d=nn.Conv1d(1,self.vectorOutDim,4)

        self.dConv1d=nn.Conv1d(1,self.vectorOutDim,4)

        self.cConv1d=nn.Conv1d(1,self.vectorOutDim,4)

        self.bufferFc=nn.Linear(1,self.scalarOutDim)

        self.leftChunkFc=nn.Linear(1,self.scalarOutDim)

        self.bitrateFc=nn.Linear(1,self.scalarOutDim)

        self.fullyConnected=nn.Linear(self.numFcInput,self.numFcOutput)

        self.outputLayer=nn.Linear(self.numFcOutput,1)

        #------------------init layer weight--------------------
        # tensorflow-1.12 uses glorot_uniform(also called xavier_uniform) to initialize weight
        # uses zero to initialize bias
        # Conv1d also use same initialize method 
        nn.init.xavier_uniform_(self.bufferFc.weight.data)
        nn.init.constant_(self.bufferFc.bias.data,0.0)
        nn.init.xavier_uniform_(self.leftChunkFc.weight.data)
        nn.init.constant_(self.leftChunkFc.bias.data,0.0)
        nn.init.xavier_uniform_(self.bitrateFc.weight.data)
        nn.init.constant_(self.bitrateFc.bias.data,0.0)
        nn.init.xavier_uniform_(self.fullyConnected.weight.data)
        nn.init.constant_(self.fullyConnected.bias.data,0.0)
        nn.init.xavier_uniform_(self.tConv1d.weight.data)
        nn.init.constant_(self.tConv1d.bias.data,0.0)
        nn.init.xavier_uniform_(self.dConv1d.weight.data)
        nn.init.constant_(self.dConv1d.bias.data,0.0)
        nn.init.xavier_normal_(self.cConv1d.weight.data)
        nn.init.constant_(self.cConv1d.bias.data,0.0)


    def forward(self,inputs):

        bitrateFcOut=F.relu(self.bitrateFc(inputs[:,0:1,-1]),inplace=True)

        bufferFcOut=F.relu(self.bufferFc(inputs[:,1:2,-1]),inplace=True)
 
        tConv1dOut=F.relu(self.tConv1d(inputs[:,2:3,:]),inplace=True)

        dConv1dOut=F.relu(self.dConv1d(inputs[:,3:4,:]),inplace=True)

        cConv1dOut=F.relu(self.cConv1d(inputs[:,4:5,:self.a_dim]),inplace=True)
       
        leftChunkFcOut=F.relu(self.leftChunkFc(inputs[:,5:6,-1]),inplace=True)

        t_flatten=tConv1dOut.view(tConv1dOut.shape[0],-1)

        d_flatten=dConv1dOut.view(dConv1dOut.shape[0],-1)

        c_flatten=cConv1dOut.view(dConv1dOut.shape[0],-1)

        fullyConnectedInput=torch.cat([bitrateFcOut,bufferFcOut,t_flatten,d_flatten,c_flatten,leftChunkFcOut],1)

        fcOutput=F.relu(self.fullyConnected(fullyConnectedInput),inplace=True)
        
        out=self.outputLayer(fcOutput)

        return out


class A3C(object):
    def __init__(self,is_central,model_type,s_dim,action_dim,actor_lr=1e-4,critic_lr=1e-3):
        self.s_dim=s_dim
        self.a_dim=action_dim
        self.discount=0.99
        self.entropy_weight=0.5
        self.entropy_eps=1e-6
        self.model_type=model_type

        self.is_central=is_central
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actorNetwork=ActorNetwork(self.s_dim,self.a_dim).to(self.device)
        if self.is_central:
            # unify default parameters for tensorflow and pytorch
            self.actorOptim=torch.optim.RMSprop(self.actorNetwork.parameters(),lr=actor_lr,alpha=0.9,eps=1e-10)
            self.actorOptim.zero_grad()
            if model_type<2:
                '''
                model==0 mean original
                model==1 mean critic_td
                model==2 mean only actor
                '''
                self.criticNetwork=CriticNetwork(self.s_dim,self.a_dim).to(self.device)
                self.criticOptim=torch.optim.RMSprop(self.criticNetwork.parameters(),lr=critic_lr,alpha=0.9,eps=1e-10)
                self.criticOptim.zero_grad()
        else:
            self.actorNetwork.eval()

        self.loss_function=nn.MSELoss()

    def getNetworkGradient(self,s_batch,a_batch,r_batch,terminal):
        s_batch=torch.cat(s_batch).to(self.device)
        a_batch=torch.LongTensor(a_batch).to(self.device)
        r_batch=torch.tensor(r_batch).to(self.device)
        R_batch=torch.zeros(r_batch.shape).to(self.device)

        R_batch[-1] = r_batch[-1]
        for t in reversed(range(r_batch.shape[0]-1)):
            R_batch[t]=r_batch[t] + self.discount*R_batch[t+1]

        if self.model_type<2:
            with torch.no_grad():
                v_batch=self.criticNetwork.forward(s_batch).squeeze().to(self.device)
            td_batch=R_batch-v_batch
        else:
            td_batch=R_batch

        probability=self.actorNetwork.forward(s_batch)
        m_probs=Categorical(probability)
        log_probs=m_probs.log_prob(a_batch)
        actor_loss=torch.sum(log_probs*(-td_batch))
        entropy_loss=-self.entropy_weight*torch.sum(m_probs.entropy())
        actor_loss=actor_loss+entropy_loss
        actor_loss.backward()


        if self.model_type<2:
            if self.model_type==0:
                # original
                critic_loss=self.loss_function(R_batch,self.criticNetwork.forward(s_batch).squeeze())
            else:
                # cricit_td
                v_batch=self.criticNetwork.forward(s_batch[:-1]).squeeze()
                next_v_batch=self.criticNetwork.forward(s_batch[1:]).squeeze().detach()
                critic_loss=self.loss_function(r_batch[:-1]+self.discount*next_v_batch,v_batch)

            critic_loss.backward()

        # use the feature of accumulating gradient in pytorch

    def actionSelect(self,stateInputs):
        if not self.is_central:
            with torch.no_grad():
                probability=self.actorNetwork.forward(stateInputs)
                m=Categorical(probability)
                action=m.sample().item()
                return action

    def hardUpdateActorNetwork(self,actor_net_params):
        for target_param,source_param in zip(self.actorNetwork.parameters(),actor_net_params):
            target_param.data.copy_(source_param.data)
 
    def updateNetwork(self):
        # use the feature of accumulating gradient in pytorch
        if self.is_central:
            self.actorOptim.step()
            self.actorOptim.zero_grad()
            if self.model_type<2:
                self.criticOptim.step()
                self.criticOptim.zero_grad()
    def getActorParam(self):
        return list(self.actorNetwork.parameters())
    def getCriticParam(self):
        return list(self.criticNetwork.parameters())


if __name__ =='__main__':
    # test maddpg in convid,ok
    SINGLE_S_LEN=19

    AGENT_NUM=1
    BATCH_SIZE=200

    S_INFO=6
    S_LEN=8
    ACTION_DIM=6

    discount=0.9

    obj=A3C(False,0,[S_INFO,S_LEN],ACTION_DIM)

    episode=3000
    for i in range(episode):
        state=torch.randn(AGENT_NUM,S_INFO,S_LEN)
        action=torch.randint(0,5,(AGENT_NUM,),dtype=torch.long)
        reward=torch.randn(AGENT_NUM)
        probability=obj.actionSelect(state)
        print(probability)
