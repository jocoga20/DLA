
    try:
        for e in range(epochs):
            model.train()
            train_loss, train_acc = adv_train(model, trainloader, optimizer, criterion, epsilon=4/255)
            model.eval()
            with torch.no_grad():
                test_loss, test_acc = eval(model, testloader, criterion)

            print(f'[{e}] L: {train_loss:.4f} / {test_loss:.4f} A: {train_acc*100:.1f} / {test_acc*100:.1f}')

            if test_acc >= best_acc + test_acc_inc:
                best_acc = test_acc
                best_state_dict = model.state_dict()
                patience = 0
            else:
                patience += 1
                if patience == max_patience:
                    print('Early stop')
                    break
    finally:
        if best_state_dict is not None:
            model_name = f'resnet50.acc{int(best_acc*100)}.pt'
            print(f'Saving {model_name}')
            torch.save(best_state_dict, model_name)